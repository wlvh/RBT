# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:21:16 2024

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym  # 使用 gymnasium 替换 gym
from gymnasium import spaces  # 使用 gymnasium 的 spaces

class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        market_dim = observation_space['market'].shape[0]
        strategies_dim = observation_space['strategies'].shape[0] * observation_space['strategies'].shape[1]
        
        # Market CNN
        self.market_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Strategies CNN
        self.strategies_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 卷积层的输出维度取决于输入的维度，在某些情况下可能需要动态计算。CNN 需要进行一次前向传播来确定扁平化后的维度，特别是当输入维度可变时。使用 with torch.no_grad() 可以防止在这个初始化过程中创建计算图，从而节省内存并提高效率。
        with torch.no_grad():
            market_sample = torch.zeros((1, 1, market_dim))
            strategies_sample = torch.zeros((1, 1, strategies_dim))
            market_flatten = self.market_cnn(market_sample).shape[1]
            strategies_flatten = self.strategies_cnn(strategies_sample).shape[1]
        
        # Linear layers
        self.market_linear = nn.Linear(market_flatten, 64)
        self.strategies_linear = nn.Linear(strategies_flatten, 64)
        
        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        market_features = self.market_linear(self.market_cnn(observations['market'].unsqueeze(1)))
        strategies_features = self.strategies_linear(self.strategies_cnn(observations['strategies'].flatten(start_dim=1).unsqueeze(1)))
        combined = torch.cat([market_features, strategies_features], dim=1)
        return self.combiner(combined)


class AutoEncoderFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(AutoEncoderFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        market_dim = observation_space['market'].shape[0]
        strategies_dim = observation_space['strategies'].shape[0] * observation_space['strategies'].shape[1]
        
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.strategies_encoder = nn.Sequential(
            nn.Linear(strategies_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        market_features = self.market_encoder(observations['market'])
        strategies_features = self.strategies_encoder(observations['strategies'].flatten(start_dim=1))
        combined = torch.cat([market_features, strategies_features], dim=1)
        return self.combiner(combined)

        # self.decoder = nn.Sequential(
        #     nn.Linear(features_dim, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, market_dim + strategies_dim)
        # )

    # def reconstruct(self, encoded):
    #     return self.decoder(encoded)

    # def encode_decode(self, observations):
    #     encoded = self.forward(observations)
    #     reconstructed = self.reconstruct(encoded)
    #     return encoded, reconstructed

    # def reconstruction_loss(self, observations):
    #     _, reconstructed = self.encode_decode(observations)
    #     original = torch.cat([observations['market'], observations['strategies'].flatten(start_dim=1)], dim=1)
    #     return F.mse_loss(reconstructed, original)

class VAEFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64, use_fat_tailed: bool = False):
        super(VAEFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        self.use_fat_tailed = use_fat_tailed
        market_dim = observation_space['market'].shape[0]
        strategies_dim = observation_space['strategies'].shape[0] * observation_space['strategies'].shape[1]
        print(f"VAEFeaturesExtractor Market dim: {market_dim}")
        print(f"VAEFeaturesExtractor Strategies dim: {strategies_dim}")       
        
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, 128),
            nn.ReLU()
        )
        self.market_mu = nn.Linear(128, 64)
        self.market_logvar = nn.Linear(128, 64)
        
        self.strategies_encoder = nn.Sequential(
            nn.Linear(strategies_dim, 128),
            nn.ReLU()
        )
        self.strategies_mu = nn.Linear(128, 64)
        self.strategies_logvar = nn.Linear(128, 64)
        
        self.combiner = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, market_dim + strategies_dim)
        )

        if use_fat_tailed:
            self.log_df = nn.Parameter(torch.zeros(1))
        else:
            self.log_df = None
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reparameterize(self, mu, logvar):
        '''
        1. logvar 的含义
        在VAE中，logvar是对数方差的输出。方差表示了潜在变量 z 的不确定性，logvar代表方差的对数形式。
        在重参数化技巧中，通过 std = exp(0.5 * logvar) 来计算标准差（std），然后用于生成潜在变量 z。
        2. 裁剪 logvar 的原因
        如果 logvar 非常接近负无穷大，exp(0.5 * logvar) 会趋近于 0，从而导致 std 非常接近 0。这可能导致数值不稳定，尤其在后续计算中，可能会引发梯度爆炸或消失的问题。
        如果 logvar 取值过大，exp(0.5 * logvar) 会非常大，导致 std 也非常大。这同样可能引发不稳定的数值计算，尤其是在反向传播中。
        3. 为什么选择 -10 和 10
        -10 的选择：对于 logvar = -10，exp(0.5 * -10) 约等于 0.0067。这是一个非常小的标准差，但仍然是可计算且不太容易引起数值不稳定的问题。这个值确保了 std 不会太接近 0，从而避免潜在的数值问题。
        10 的选择：对于 logvar = 10，exp(0.5 * 10) 约等于 148.41。这是一个相对较大的标准差，通常可以容纳大部分数据的变异性，同时避免了标准差变得过大导致的数值不稳定。
        这些值在许多实际应用中被证明是合理的，但并不是唯一选择。它们是用于确保 logvar 在合理的范围内波动的经验性选择
        '''
        # print(f"Original logvar range: {logvar.min().item()} to {logvar.max().item()}")
        logvar = torch.clamp(logvar, min=-20, max=20)  # 对 logvar 进行裁剪
        # print(f"Clamped logvar range: {logvar.min().item()} to {logvar.max().item()}")
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, observations):
        # print("Market observation shape:", observations['market'].shape)
        # print("Strategies observation shape:", observations['strategies'].shape)
        market_h = self.market_encoder(observations['market'])
        # print("Market encoded range:", market_h.min().item(), "to", market_h.max().item())
        market_mu = self.market_mu(market_h)
        market_logvar = self.market_logvar(market_h)
        market_z = self.reparameterize(market_mu, market_logvar)
        
        strategies_flat = observations['strategies'].flatten(start_dim=1)
        strategies_h = self.strategies_encoder(strategies_flat)
        # print("Strategies encoded range:", strategies_h.min().item(), "to", strategies_h.max().item())
        strategies_mu = self.strategies_mu(strategies_h)
        strategies_logvar = self.strategies_logvar(strategies_h)
        strategies_z = self.reparameterize(strategies_mu, strategies_logvar)
        # print(f"Market logvar range: {market_logvar.min().item()} to {market_logvar.max().item()}")
        # print(f"Strategies logvar range: {strategies_logvar.min().item()} to {strategies_logvar.max().item()}")
        # print("Market encoded shape:", market_h.shape)
        # print("Strategies encoded shape:", strategies_h.shape)
        # print("market_Mu shape:", market_mu.shape)
        # print("market_Logvar shape:", market_logvar.shape)  
        # print("strategies_Mu shape:", strategies_mu.shape)
        # print("strategies_Logvar shape:", strategies_logvar.shape) 
        # print("market_Z shape:", market_z.shape)
        # print("strategies_Z shape:", strategies_z.shape)
        
        combined = torch.cat([market_z, strategies_z], dim=1)
        # print("Combined range:", combined.min().item(), "to", combined.max().item())
        # 添加一个归一化步骤
        combined = F.normalize(combined, p=2, dim=1)
        # print("Normalized combined range:", combined.min().item(), "to", combined.max().item())
        result = self.combiner(combined)
        # print("Final output range:", result.min().item(), "to", result.max().item())        
        
        
        combined = torch.cat([market_z, strategies_z], dim=1)
        print("Combined range:", combined.min().item(), "to", combined.max().item())
        result = self.combiner(combined)
        print("Final output range:", result.min().item(), "to", result.max().item())
        return result

    def encode(self, observations):
        # 获取市场观察数据并转换为张量
        market_obs = torch.from_numpy(observations['market']).float().to(self.device)
        # 获取策略观察数据并转换为张量
        strategies_obs = torch.from_numpy(observations['strategies']).float().to(self.device)
        # print(f"VAE encode market_obs shape: {market_obs.shape}")
        # print(f"VAE encode strategies_obs shape: {strategies_obs.shape}") 
        # 编码市场观察数据
        market_h = self.market_encoder(market_obs)
        # print(f"VAE encode market_h shape: {market_h.shape}")
        
        # 调整 market_h 形状，使其具有 batch 维度
        market_h = market_h.view(1, -1)
        
        # 将策略观察数据展平成 (1, 4104)
        strategies_obs_flat = strategies_obs.view(1, -1)
        # print(f"VAE encode strategies_obs_flat shape: {strategies_obs_flat.shape}")
        # 编码策略观察数据
        strategies_h = self.strategies_encoder(strategies_obs_flat)
        # print(f"VAE encode strategies_h shape: {strategies_h.shape}")        
        
        # 获取均值和对数方差
        mu = torch.cat([self.market_mu(market_h), self.strategies_mu(strategies_h)], dim=1)
        logvar = torch.cat([self.market_logvar(market_h), self.strategies_logvar(strategies_h)], dim=1)
        
        logvar = torch.clamp(logvar, min=-20, max=20)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def get_df(self):
        if self.use_fat_tailed:
            return torch.exp(self.log_df)
        else:
            return None

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        计算近似后验分布q(正态分布)和先验分布p(可能是t分布)之间的KL散度。
    
        Args:
            mu (torch.Tensor): 近似后验分布的均值。
            logvar (torch.Tensor): 近似后验分布的对数方差。
    
        Returns:
            torch.Tensor: 计算得到的KL散度。
    
        Raises:
            ValueError: 如果use_fat_tailed为True但get_df()返回无效值。
        """
        print("Final kl_divergence logvar range:", logvar.min().item(), "to", logvar.max().item())  
        print("Final kl_divergence mu range:", mu.min().item(), "to", mu.max().item())  
        
        var = torch.exp(logvar)
        
        if self.use_fat_tailed:
            df = self.get_df()
            if df is None or df <= 0:
                raise ValueError("Invalid degrees of freedom for fat-tailed distribution.")
            
            # KL(N(mu,var) || T(df,0,1))
            kl = 0.5 * (torch.log(df) + df + mu**2 + var * (df - torch.tensor(2.0)) / 2
                        - torch.log(var) - 1
                        - torch.digamma(df/2) - torch.log(torch.tensor(2.0)))
            kl = kl + torch.lgamma(df/2) - torch.lgamma((df+torch.tensor(1.0))/2)
        else:
            # KL(N(mu,var) || N(0,1))
            kl = 0.5 * (mu**2 + var - logvar - 1)
        
        return kl.sum(dim=-1)

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = self.kl_divergence(mu, logvar)
        return BCE + KLD

    def sample(self, num_samples):
        if self.use_fat_tailed:
            df = self.get_df()
            p = dist.StudentT(df, loc=0, scale=1)
        else:
            p = dist.Normal(0, 1)
        z = p.sample((num_samples, self.combiner[0].in_features))
        return self.decode(z)

    def monitor_df(self):
        '''
        肥尾分布的自由度参数之所以对调优模型非常有帮助，主要有以下几个原因：
        捕捉市场异常：它能帮助模型更好地捕捉和预测金融市场中的极端事件。
        动态风险管理：通过监控自由度参数，可以实现动态的风险管理策略。
        模型适应性：自由度参数的变化反映了模型对市场条件变化的适应。
        性能评估：它提供了一个额外的指标来评估模型的表现和适用性。
        策略优化：基于自由度参数的变化，可以优化交易策略，在不同的市场环境中采取不同的行动。
        早期预警：自由度参数的突然变化可能预示着市场状态的重大转变，为及时调整策略提供了信号。
        '''
        if self.use_fat_tailed:
            return self.get_df().item()
        else:
            return None


















