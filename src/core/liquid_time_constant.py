import torch
import torch.nn as nn
import torch.amp

from typing import Optional, Tuple


class LiquidTimeConstantNetwork(nn.Module):
    def __init__(
        self,
        field_dim: int = 4096,
        hidden_dim: int = 4096,
        tau_max: float = 1.0,
        noise_std: float = 0.01,
        state_clip: float = 5.0,
        use_fp16: bool = True,
        use_amp: bool = True,
        use_cuda_graph: bool = False,
        dt: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.field_dim = field_dim
        self.hidden_dim = hidden_dim
        self.tau_max = tau_max
        self.noise_std = noise_std
        self.state_clip = state_clip
        self.dt = dt
        self.use_fp16 = use_fp16 and device != "cpu"
        self.use_amp = use_amp and device != "cpu"
        # CUDA Graph仅适用于无AMP、固定dt的场景
        self.use_cuda_graph = use_cuda_graph and device == "cuda" and not self.use_amp
        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim)
        )

        tau_hidden = min(256, field_dim // 4)
        self.tau_net = nn.Sequential(
            nn.Linear(field_dim, tau_hidden),
            nn.GELU(),
            nn.Linear(tau_hidden, field_dim)
        )

        self.field_state = nn.Parameter(
            torch.randn(field_dim) * 0.1,
            requires_grad=False
        )

        self.to(device)

        if self.use_fp16 and not self.use_amp:
            self.mlp = self.mlp.half()
            self.tau_net = self.tau_net.half()
            self.field_state.data = self.field_state.data.half()
        
        if self.use_amp:
            self.amp_scaler = torch.amp.GradScaler('cuda')
        else:
            self.amp_scaler = None
        
        if self.use_cuda_graph:
            self._cuda_graph = None

    def compute_time_constants(self, h: torch.Tensor) -> torch.Tensor:
        tau_linear = self.tau_net(h)
        tau = torch.sigmoid(tau_linear) * self.tau_max
        return tau

    def evolve(
        self,
        h: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        dt: float = 0.1,
        return_tau: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        if h is None:
            h = self.field_state.data

        if u is None:
            # 无外部输入时尝试使用CUDA Graph加速
            if self.use_cuda_graph and dt == self.dt:
                if self._cuda_graph is None:
                    self._capture_cuda_graph(h)
                self._static_input.copy_(h)
                self._cuda_graph.replay()
                # 在Graph外添加噪声（Graph只能捕获确定性计算）
                noise = torch.randn(self.field_dim, device=self.device, dtype=h.dtype) * self.noise_std
                h_new = torch.clamp(self._static_output + noise, -self.state_clip, self.state_clip)
                # CUDA Graph已包含残差连接 h + dh，此处仅加噪声
                self.field_state.data = h_new
                if return_tau:
                    tau = self.compute_time_constants(h)
                    return h_new, tau
                return h_new

            u = torch.zeros(self.field_dim, device=self.device, dtype=h.dtype)

        if self.use_amp and self.device != "cpu":
            with torch.amp.autocast('cuda'):
                f_h = self.mlp(h + u)
                tau = self.compute_time_constants(h)
        else:
            f_h = self.mlp(h + u)
            tau = self.compute_time_constants(h)

        noise = torch.randn(self.field_dim, device=self.device, dtype=h.dtype) * self.noise_std

        dh = f_h * tau * dt
        h_new = h + dh + noise

        h_new = torch.clamp(h_new, -self.state_clip, self.state_clip)

        self.field_state.data = h_new

        if return_tau:
            return h_new, tau
        return h_new

    def _capture_cuda_graph(self, h: torch.Tensor):
        """捕获CUDA Graph用于加速固定计算图（仅捕获确定性部分）"""
        self._static_input = torch.zeros(self.field_dim, device=self.device, dtype=h.dtype)
        self._static_input.copy_(h)
        
        # Warmup (CUDA Graph需要先warmup)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._static_output = self._compute_dh(self._static_input)
        torch.cuda.current_stream().wait_stream(s)
        
        # 捕获
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._static_output = self._compute_dh(self._static_input)
    
    def _compute_dh(self, h: torch.Tensor) -> torch.Tensor:
        """确定性演化步（含残差连接，无噪声，可被CUDA Graph捕获）"""
        f_h = self.mlp(h)
        tau = torch.sigmoid(self.tau_net(h)) * self.tau_max
        return h + f_h * tau * self.dt

    def reset_state(self, init_scale: float = 0.1):
        dtype = self.field_state.data.dtype
        self.field_state.data = torch.randn(self.field_dim, device=self.device, dtype=dtype) * init_scale

    def get_state(self) -> torch.Tensor:
        return self.field_state.data.clone()

    def set_state(self, h: torch.Tensor):
        self.field_state.data = h.to(device=self.device, dtype=self.field_state.data.dtype)
