import time

import torch
from torch import nn

from pearl.data import BallData, PlayerData, BoostData, GameInfo


class CarballTransformer(nn.Module):
    def __init__(self, *, dim: int, num_layers: int, num_heads=None, ff_dim=None, activation_fn=nn.GELU,
                 include_game_info=False):
        super(CarballTransformer, self).__init__()

        if num_heads is None:
            num_heads = dim // 64
        if ff_dim is None:
            ff_dim = dim * 4

        self.game_columns = len(GameInfo) - 1
        self.ball_columns = len(BallData) - 1
        self.player_columns = len(PlayerData) - 1
        self.boost_columns = len(BoostData) - 1

        if include_game_info:
            self.game_embedding = nn.Sequential(
                nn.Linear(self.game_columns, dim),
                activation_fn(),
                nn.Linear(dim, dim)
            )
        else:
            self.game_embedding = None
        self.ball_embedding = nn.Sequential(
            nn.Linear(self.ball_columns, dim),
            activation_fn(),
            nn.Linear(dim, dim)
        )
        self.player_embedding = nn.Sequential(
            nn.Linear(self.player_columns, dim),
            activation_fn(),
            nn.Linear(dim, dim)
        )

        self.boost_embedding = nn.Sequential(
            nn.Linear(self.boost_columns, dim),
            activation_fn(),
            nn.Linear(dim, dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=ff_dim,
                                                   activation=activation_fn(),
                                                   batch_first=True,
                                                   norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, game_info, ball_data, player_data, boost_data):
        # Shapes:
        # game_info: (batch_size, num_columns)
        # ball_data: (batch_size, num_balls, num_columns)
        # player_data: (batch_size, num_players, num_columns)
        # boost_data: (batch_size, num_boosts, num_columns)

        if self.game_embedding is None or game_info is None:
            game_info = torch.zeros((ball_data.shape[0], 0, self.game_columns), device=ball_data.device)
        else:
            game_info = game_info.unsqueeze(1)
        ignore_mask = torch.cat([
            game_info[:, :, GameInfo.IGNORE],
            ball_data[:, :, BallData.IGNORE],
            player_data[:, :, PlayerData.IGNORE],
            boost_data[:, :, BoostData.IGNORE]
        ], dim=1)
        entities = []
        if self.game_embedding is not None:
            game_embedded = self.game_embedding(game_info[:, :, 1:])
            entities.append(game_embedded)
        ball_embedded = self.ball_embedding(ball_data[:, :, 1:])
        player_embedded = self.player_embedding(player_data[:, :, 1:])
        boost_embedded = self.boost_embedding(boost_data[:, :, 1:])

        entities += [ball_embedded, player_embedded, boost_embedded]
        entities_out = torch.cat(entities, dim=1)
        entities_out = self.encoder(entities_out, src_key_padding_mask=ignore_mask)

        results = torch.split(entities_out, [e.shape[1] for e in entities], dim=1)

        return results


class NextGoalPredictor(nn.Module):
    def __init__(self, transformer: CarballTransformer, include_ties=False):
        super(NextGoalPredictor, self).__init__()

        self.transformer = transformer
        self.linear = nn.Linear(transformer.encoder.layers[-1].linear2.out_features, 2 + include_ties)

    def forward(self, game_info, ball_data, player_data, boost_data):
        out = self.transformer(game_info, ball_data, player_data, boost_data)
        if len(out) == 4:
            game_out, ball_out, player_out, boost_out = out
        else:
            ball_out, player_out, boost_out = out
        out = self.linear(ball_out)
        return out.squeeze((1, 2))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for (size, dim, layers) in (("tiny", 128, 2), ("small", 256, 4), ("medium", 512, 8),
                                ("base", 768, 12), ("large", 1024, 24), ("xlarge", 2048, 24)):
        print(f"Size: '{size}', dims: {dim}, layers: {layers}")
        transformer = CarballTransformer(dim, layers)
        predictor = NextGoalPredictor(transformer).to(device)
        print(f"\tParams: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):_}")

        bs = 64
        inp = (torch.randn(bs, 1, len(BallData)),
               torch.randn(bs, 5, len(PlayerData)),
               torch.randn(bs, len(BoostData)))
        inp = tuple(x.to(device) for x in inp)

        t0 = time.perf_counter()
        out = predictor(*inp)
        t1 = time.perf_counter()
        print(f"\tTime: {t1 - t0:.4f}s")

        loss_fn = nn.BCEWithLogitsLoss()

        target = torch.randint(2, (bs,)).float().to(device)
        print(f"\tLoss: {loss_fn(out, target):.4f}")

        debug = True
