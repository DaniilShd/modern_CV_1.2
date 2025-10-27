import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from transformers import ViTModel, ViTImageProcessor
import torch.nn.functional as F

class AttentionAnalyzer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ViTModel.from_pretrained(
            model_path,
            output_attentions=True,  # ВАЖНО: включаем attention
            add_pooling_layer=False
        ).to(device)
        self.model.eval()
        
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.patch_size = self.model.config.patch_size
        self.num_heads = self.model.config.num_attention_heads
        self.num_layers = self.model.config.num_hidden_layers
    
    def preprocess_image(self, image):
        """Препроцессинг изображения"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        inputs = self.processor(image, return_tensors='pt')
        return inputs.to(self.device), image
    
    def get_attention_maps(self, image):
        """Извлечение attention карт"""
        inputs, original_image = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # [layers, batch, heads, tokens, tokens]
        
        return attentions, inputs, original_image
    
    def visualize_cls_attention(self, image, layer_idx=0, head_idx=0):
        """Визуализация внимания от CLS токена"""
        attentions, inputs, original_img = self.get_attention_maps(image)
        
        # Attention от CLS токена к патчам
        cls_attention = attentions[layer_idx][0, head_idx, 0, 1:]  # [num_patches]
        
        # Преобразуем в heatmap
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        
        return self._overlay_heatmap(original_img, attention_map)
    
    def attention_rollout(self, image):
        """Attention Rollout через все слои"""
        attentions, inputs, original_img = self.get_attention_maps(image)
        
        # Инициализируем rollout матрицу
        batch_size, num_heads, num_tokens, _ = attentions[0].shape
        rollout = torch.eye(num_tokens).to(self.device)
        
        # Применяем rollout через все слои
        for layer_attention in attentions:
            # Усредняем по головам
            layer_attention_avg = layer_attention.mean(dim=1)  # [batch, tokens, tokens]
            rollout = torch.matmul(layer_attention_avg[0], rollout)
        
        # Берем attention от CLS токена
        cls_attention_rollout = rollout[0, 1:].cpu().numpy()
        
        grid_size = int(np.sqrt(cls_attention_rollout.shape[0]))
        attention_map = cls_attention_rollout.reshape(grid_size, grid_size)
        
        return self._overlay_heatmap(original_img, attention_map)
    
    def compare_heads_layers(self, image, save_path=None):
        """Сравнение attention между головами и слоями"""
        attentions, _, original_img = self.get_attention_maps(image)
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Визуализируем разные слои (последние 12)
        start_layer = max(0, self.num_layers - 12)
        
        for i, layer_idx in enumerate(range(start_layer, self.num_layers)):
            if i >= len(axes):
                break
                
            # Усредняем по головам
            layer_attention = attentions[layer_idx][0].mean(dim=0)  # [tokens, tokens]
            cls_attention = layer_attention[0, 1:].cpu().numpy()
            
            grid_size = int(np.sqrt(cls_attention.shape[0]))
            attention_map = cls_attention.reshape(grid_size, grid_size)
            
            axes[i].imshow(attention_map, cmap='hot')
            axes[i].set_title(f'Layer {layer_idx}')
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_head_diversity(self, image):
        """Анализ разнообразия attention heads"""
        attentions, _, _ = self.get_attention_maps(image)
        
        diversity_scores = []
        for layer_idx, layer_attention in enumerate(attentions):
            # [batch, heads, tokens, tokens]
            head_attentions = layer_attention[0]  # [heads, tokens, tokens]
            
            # Считаем разнообразие между головами
            cls_attentions = head_attentions[:, 0, 1:]  # [heads, patches]
            correlations = []
            
            for i in range(self.num_heads):
                for j in range(i+1, self.num_heads):
                    corr = F.cosine_similarity(
                        cls_attentions[i], cls_attentions[j], dim=0
                    )
                    correlations.append(corr.item())
            
            avg_correlation = np.mean(correlations)
            diversity_scores.append((layer_idx, 1 - avg_correlation))
        
        return diversity_scores
    
    def interpolate_positional_embeddings(self, new_size=(128, 128)):
        """Интерполяция позиционных эмбеддингов для нового размера"""
        old_embeddings = self.model.embeddings.position_embeddings
        
        # CLS token + патчи
        old_num_tokens = 1 + (224 // self.patch_size) ** 2
        new_num_tokens = 1 + (new_size[0] // self.patch_size) ** 2
        
        if new_num_tokens != old_num_tokens:
            # Интерполяция 2D позиционных эмбеддингов
            old_grid_size = int(np.sqrt(old_num_tokens - 1))
            new_grid_size = int(np.sqrt(new_num_tokens - 1))
            
            # Переформатируем в 2D grid (исключая CLS token)
            old_pos_emb = old_embeddings[1:].reshape(1, old_grid_size, old_grid_size, -1)
            old_pos_emb = old_pos_emb.permute(0, 3, 1, 2)  # [1, dim, grid, grid]
            
            # Интерполяция
            new_pos_emb = F.interpolate(
                old_pos_emb, 
                size=(new_grid_size, new_grid_size), 
                mode='bicubic', 
                align_corners=False
            )
            
            new_pos_emb = new_pos_emb.permute(0, 2, 3, 1).reshape(-1, new_pos_emb.shape[1])
            
            # Добавляем CLS token embedding
            cls_emb = old_embeddings[0:1]
            new_embeddings = torch.cat([cls_emb, new_pos_emb], dim=0)
            
            return new_embeddings
        
        return old_embeddings
    
    def _overlay_heatmap(self, image, attention_map):
        """Наложение heatmap на изображение"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Оригинальное изображение
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        im = ax2.imshow(attention_map, cmap='hot')
        ax2.set_title('Attention Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Наложение
        img_array = np.array(image)
        if img_array.shape[-1] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        heatmap = cv2.resize(attention_map, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), 0.6,
            heatmap, 0.4, 0
        )
        
        ax3.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        return fig