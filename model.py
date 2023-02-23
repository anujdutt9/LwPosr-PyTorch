# Import Dependencies
import torch
import torch.nn as nn


class UpperStream(nn.Module):
    def __init__(self, input_ch: int = 3):
        """Class containing Stages in Upper Stream of the Model

        Args:
            input_ch (int): Number of channels in the input. Defaults to 3.
        """
        super().__init__()
        
        # Transformer Encoder Layer (as per paper)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=64)
        
        # Block-1
        self.stage1 = nn.Sequential(
            self.DWSepConv2DLayer(in_ch=input_ch, out_ch=16),
            nn.MaxPool2d(kernel_size=2),
            self.DWSepConv2DLayer(in_ch=16, out_ch=32),
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Block-2
        self.stage2 = nn.Sequential(
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
        )
        
        self.stage2_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
        self.stage2_pool = nn.MaxPool2d(kernel_size=2)
        
        # Block-3
        self.stage3 = nn.Sequential(
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
        )
        
        self.stage3_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
    
    # Depthwise Separable Conv2D
    def DWSepConv2DLayer(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Deptwise Separable Conv2D with ReLU Activation (as per paper)

        Args:
            in_ch (int): Number of Input Channels
            out_ch (int): Number of Output Channels

        Returns:
            nn.Sequential: Instance of Depthwise Separable Conv2D
        """
        depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, groups=in_ch)
        point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        depthwise_separable_conv2d = nn.Sequential(
            depth_conv,
            point_conv,
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        return depthwise_separable_conv2d
    
    def forward(self, x: torch.Tensor) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        """_summary_

        Args:
            x (torch.Tensor): Input Image Batch Tensor of shape (B, C, H, W)

        Returns:
            list(torch.Tensor) : Outputs from every stage for input to Mixture Blocks
        """
        x = self.stage1(x)    # out1
        x1 = x
        
        x = self.stage2(x)
        # Reshape Transform - (B, C, H, W) -> (B, C, A) -> (B, A, C), where A = H x W
        b, c, h, w = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1)
        x = self.stage2_transformer(x)
        x = x.permute(0,2,1)     # (B, A, C) -> (B, C, A)
        x = x.view(b, c, h, w)   # (B, C, A) -> (B, C, H, W)
        x = self.stage2_pool(x)    # out2
        x2 = x
        
        x = self.stage3(x)
        # Reshape Transform - (B, C, H, W) -> (B, C, A) -> (B, A, C), where A = H x W
        b, c, h, w = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1)
        x = self.stage3_transformer(x)
        x = x.permute(0,2,1)     # (B, A, C) -> (B, C, A)
        x = x.view(b, c, h, w)   # (B, C, A) -> (B, C, H, W)
        
        return (x1, x2, x)


class LowerStream(nn.Module):
    def __init__(self, input_ch: int = 3):
        """Class containing Stages in Lower Stream of the Model

        Args:
            input_ch (int): Number of channels in the input. Defaults to 3.
        """
        super().__init__()
        
        # Transformer Encoder Layer - as per paper
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=64)
        
        # Block-1
        self.stage1 = nn.Sequential(
            self.DWSepConv2DLayer(in_ch=input_ch, out_ch=16),
            nn.AvgPool2d(kernel_size=2),
            self.DWSepConv2DLayer(in_ch=16, out_ch=32),
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
            nn.AvgPool2d(kernel_size=2),
        )
        
        # Block-2
        self.stage2 = nn.Sequential(
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
        )
        
        self.stage2_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
        self.stage2_pool = nn.AvgPool2d(kernel_size=2)
        
        # Block-3
        self.stage3 = nn.Sequential(
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
            self.DWSepConv2DLayer(in_ch=32, out_ch=32),
        )
        
        self.stage3_transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
    
    # Depthwise Separable Conv2D
    def DWSepConv2DLayer(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Deptwise Separable Conv2D with Tanh Activation (as per paper)

        Args:
            in_ch (int): Number of Input Channels
            out_ch (int): Number of Output Channels

        Returns:
            nn.Sequential: Instance of Depthwise Separable Conv2D
        """
        depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, groups=in_ch)
        point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        depthwise_separable_conv2d = nn.Sequential(
            depth_conv,
            point_conv,
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        return depthwise_separable_conv2d
    
    def forward(self, x: torch.Tensor) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        """_summary_

        Args:
            x (torch.Tensor): Input Image Batch Tensor of shape (B, C, H, W)

        Returns:
            list(torch.Tensor) : Outputs from every stage for input to Mixture Blocks
        """
        x = self.stage1(x)
        x1 = x
        
        x = self.stage2(x)
        # Reshape Transform - (B, C, H, W) -> (B, C, A) -> (B, A, C), where A = H x W
        b, c, h, w = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1)
        x = self.stage2_transformer(x)
        x = x.permute(0,2,1)     # (B, A, C) -> (B, C, A)
        x = x.view(b, c, h, w)   # (B, C, A) -> (B, C, H, W)
        x = self.stage2_pool(x)
        x2 = x
        
        x = self.stage3(x)
        # Reshape Transform - (B, C, H, W) -> (B, C, A) -> (B, A, C), where A = H x W
        b, c, h, w = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1) 
        x = self.stage3_transformer(x)
        x = x.permute(0,2,1)     # (B, A, C) -> (B, C, A)
        x = x.view(b, c, h, w)   # (B, C, A) -> (B, C, H, W)
        
        return (x1, x2, x)


class LwPosr(nn.Module):
    def __init__(self, input_ch: int = 3, output_ch: int = 3):
        """_summary_

        Args:
            input_ch (int, optional): Number of Input Channels. Defaults to 3.
            output_ch (int, optional): Number of Output Channels. Defaults to 3.
        """
        
        super().__init__()
        
        # Upper Stream
        self.upper_stage = UpperStream(input_ch=input_ch)
        
        # Lower Stream
        self.lower_stage = LowerStream(input_ch=input_ch)
        
        # Avg Pool for Stage-1 Output, since all Mixture Block Outputs have same dimension (as per paper)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        
        # Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.Flatten(),
            nn.Linear(in_features=16*56*56, out_features=output_ch),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): Input Image Batch Tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Single Tensor containing weighted mean of all mixture block outputs
        """
        x1_upper, x2_upper, x3_upper = self.upper_stage(x)
        x1_lower, x2_lower, x3_lower = self.lower_stage(x)
        
        # Mixture Block 1
        mixture_block1 = self.avg_pool(torch.mul(x1_upper, x1_lower))  # First Input to Pred Head
        # Mixture Block 2
        mixture_block2 = torch.mul(x2_upper, x2_lower)  # Second Input to Pred Head
        # Mixture Blcok 3
        mixture_block3 = torch.mul(x3_upper, x3_lower)  # Third Input to Pred Head
        
        # Prediction Head
        # Individual Pose Predictions from all 3 Mixture Blocks
        mb1_out = self.prediction_head(mixture_block1)
        mb2_out = self.prediction_head(mixture_block2)
        mb3_out = self.prediction_head(mixture_block3)
        
        # Final Output - Weighted Mean of all Pose Values from 3 Mixture Blocks
        # Weight values - from paper
        weighted_mb1_out = torch.mul(mb1_out, 0.5)
        weighted_mb2_out = torch.mul(mb2_out, 0.5)
        weighted_mb3_out = torch.mul(mb3_out, 2)
        
        # Weighted Mean of all three Mixture Block Outputs
        out = torch.mean((weighted_mb1_out + weighted_mb2_out + weighted_mb3_out), 0)
        
        return out


# TEST
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Random Input Tensor
    x = torch.rand(1, 3, 450, 450).to(device)
    print(f"Input Tensor Shape: {x.shape}")

    # Upper Stage
    upper_stage = UpperStream(x.shape[1]).to(device)
    x1_upper, x2_upper, x3_upper = upper_stage(x)
    print(f"Upper Stage - x1_upper: {x1_upper.shape}, x2_upper: {x2_upper.shape}, x3_upper: {x3_upper.shape}")

    # Lower Stage
    lower_stage = LowerStream(x.shape[1]).to(device)
    x1_lower, x2_lower, x3_lower = lower_stage(x)
    print(f"Lower Stage - x1_lower: {x1_lower.shape}, x2_lower: {x2_lower.shape}, x3_lower: {x3_lower.shape}")

    # Full Model
    model = LwPosr(input_ch=x.shape[1], output_ch=3).to(device)
    output = model(x)
    print(f"LwPosr - output: {output.shape}")

    print("\nModel Summary\n")
    print(model)

    torch.save(model.state_dict(), "LwPosr.pt")
