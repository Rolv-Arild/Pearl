import time

import torch
from torch import nn

from data import BallData, PlayerData, BoostData


class CarballTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads=None, ff_dim=None, activation_fn=nn.GELU):
        super(CarballTransformer, self).__init__()

        if num_heads is None:
            num_heads = dim // 64
        if ff_dim is None:
            ff_dim = dim * 4

        self.ball_columns = len(BallData) - 1
        self.player_columns = len(PlayerData) - 1
        self.boost_columns = len(BoostData) - 1

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

    def forward(self, ball_data, player_data, boost_data):
        # Shapes:
        # ball_data: (batch_size, num_balls, num_columns)
        # player_data: (batch_size, num_players, num_columns)
        # boost_data: (batch_size, num_boosts, num_columns)

        ignore_mask = torch.cat([
            ball_data[:, :, BallData.IGNORE.value],
            player_data[:, :, PlayerData.IGNORE.value],
            boost_data[:, :, BoostData.IGNORE.value]
        ], dim=1)
        ball_embedded = self.ball_embedding(ball_data[:, :, 1:])
        player_embedded = self.player_embedding(player_data[:, :, 1:])
        boost_embedded = self.boost_embedding(boost_data[:, :, 1:])

        entities = torch.cat([ball_embedded, player_embedded, boost_embedded], dim=1)
        entities = self.encoder(entities, src_key_padding_mask=ignore_mask)

        ball_out = entities[:, :ball_data.shape[1]]
        player_out = entities[:, ball_data.shape[1]:ball_data.shape[1] + player_data.shape[1]]
        boost_out = entities[:, ball_data.shape[1] + player_data.shape[1]:]

        return ball_out, player_out, boost_out


class NextGoalPredictor(nn.Module):
    def __init__(self, transformer: CarballTransformer):
        super(NextGoalPredictor, self).__init__()

        self.transformer = transformer
        self.linear = nn.Linear(transformer.encoder.layers[-1].linear2.out_features, 1)

    def forward(self, ball_data, player_data, boost_data):
        ball_out, player_out, boost_out = self.transformer(ball_data, player_data, boost_data)
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
