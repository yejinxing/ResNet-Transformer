import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradient = None
        self.activation = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def get_heatmap(self, input_image, class_idx=None):
        # 前向传播
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot)

        # 计算权重并生成热力图
        weights = torch.mean(self.gradient, dim=(2, 3))
        heatmap = torch.sum(weights[:, :, None, None] * self.activation, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        return heatmap

    @staticmethod
    def overlay_heatmap(img, heatmap):
        heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img.cpu().numpy().transpose(1,2,0) * 255 * 0.5
        return superimposed_img.astype('uint8')

    def generate_all_classes(self, inputs, class_names):
        self.model.zero_grad()
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 1)

        plt.figure(figsize=(15, 10))
        for cls_idx in range(len(class_names)):
            self.model.zero_grad()
            one_hot = torch.zeros_like(outputs)
            one_hot[:, cls_idx] = 1
            outputs.backward(gradient=one_hot, retain_graph=True)

            gradients = self.gradient[0].cpu().numpy()
            activations = self.activation[0].cpu().numpy()
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.sum(weights[:, None, None] * activations, axis=0)
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, inputs.shape[2:][::-1])
            cam = cam / cam.max()

            ax = plt.subplot(3, 4, cls_idx+1)
            img = inputs[0].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype('uint8')
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(img, 0.5, heatmap, 0.4, 0)
            
            plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
            plt.title(class_names[cls_idx])
            plt.axis('off')
            plt.colorbar(mappable=ax.images[0], ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return plt