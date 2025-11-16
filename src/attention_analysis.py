import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from transformers import ViTModel, ViTImageProcessor, ViTConfig
import torch.nn.functional as F

class AttentionAnalyzer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Явная загрузка конфигурации
        config = ViTConfig.from_pretrained(
            model_path,
            output_attentions=True,
            add_pooling_layer=False
        )
        
        self.model = ViTModel.from_pretrained(
            model_path,
            config=config
        ).to(device)
        self.model.eval()
        
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.patch_size = self.model.config.patch_size
        self.num_heads = self.model.config.num_attention_heads
        self.num_layers = self.model.config.num_hidden_layers
        
        print(f"=== МОДЕЛЬ ЗАГРУЖЕНА ===")
        print(f"Patch size: {self.patch_size}")
        print(f"Количество слоев: {self.num_layers}")
        print(f"Количество голов внимания: {self.num_heads}")
        print(f"Ожидаемое количество токенов: {1 + (224 // self.patch_size) ** 2}")
    
    def preprocess_image(self, image):
        """Препроцессинг изображения с диагностикой"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        inputs = self.processor(
            image, 
            return_tensors='pt',
            size={"height": 224, "width": 224}  # Явное указание размера
        )
        
        print(f"Входные данные shape: {inputs['pixel_values'].shape}")
        return inputs.to(self.device), image
    
    def get_attention_maps(self, image):
        """Извлечение attention карт с расширенной диагностикой"""
        inputs, original_image = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # [layers, batch, heads, tokens, tokens]
        
        # ДИАГНОСТИКА
        if attentions is not None and len(attentions) > 0:
            attention_shape = attentions[0].shape
            print(f"=== ATTENTION ДИАГНОСТИКА ===")
            print(f"Количество слоев: {len(attentions)}")
            print(f"Форма attention матрицы: {attention_shape}")
            print(f"Количество токенов: {attention_shape[-1]}")
            
            # Проверяем первый слой
            first_layer = attentions[0][0]  # [heads, tokens, tokens]
            print(f"Форма первого слоя: {first_layer.shape}")
            
            # Сумма внимания по головам (должна быть ~1.0)
            attention_sum = first_layer[0].sum(dim=-1)
            print(f"Сумма внимания для первого токена: {attention_sum[0]:.3f}")
        else:
            print("ВНИМАНИЕ: Attention карты не извлечены!")
        
        return attentions, inputs, original_image
    
    def visualize_cls_attention(self, image, layer_idx=0, head_idx=0):
        """Визуализация внимания от CLS токена с проверкой"""
        attentions, inputs, original_img = self.get_attention_maps(image)
        
        if attentions is None or len(attentions) == 0:
            print("Ошибка: Нет attention карт для визуализации")
            return None
        
        num_tokens = attentions[layer_idx].shape[-1]
        print(f"Визуализация слоя {layer_idx}, голова {head_idx}, токенов: {num_tokens}")
        
        # Проверяем индексы
        if layer_idx >= len(attentions):
            print(f"Ошибка: layer_idx {layer_idx} превышает количество слоев {len(attentions)}")
            return None
        
        if head_idx >= self.num_heads:
            print(f"Ошибка: head_idx {head_idx} превышает количество голов {self.num_heads}")
            return None
        
        # Attention от CLS токена к патчам
        try:
            if num_tokens > 1:
                cls_attention = attentions[layer_idx][0, head_idx, 0, 1:]  # [num_patches]
                
                # Преобразуем в heatmap
                grid_size = int(np.sqrt(cls_attention.shape[0]))
                if grid_size * grid_size == cls_attention.shape[0]:
                    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
                    return self._overlay_heatmap(original_img, attention_map)
                else:
                    print(f"Ошибка: невозможно преобразовать {cls_attention.shape[0]} патчей в квадратную матрицу")
            else:
                print("Ошибка: недостаточно токенов для визуализации")
        except Exception as e:
            print(f"Ошибка при визуализации: {e}")
        
        return None
    
    def attention_rollout(self, image, discard_ratio=0.9):
        """Улучшенный Attention Rollout через все слои"""
        attentions, inputs, original_img = self.get_attention_maps(image)
        
        if attentions is None or len(attentions) == 0:
            print("Ошибка: Нет attention карт для rollout")
            return None
        
        # Инициализируем rollout матрицу
        batch_size, num_heads, num_tokens, _ = attentions[0].shape
        rollout = torch.eye(num_tokens).to(self.device)
        
        # Применяем rollout через все слои
        for layer_attention in attentions:
            # Усредняем по головам
            layer_attention_avg = layer_attention.mean(dim=1)  # [batch, tokens, tokens]
            
            # Улучшение: добавляем тождественную матрицу
            I = torch.eye(num_tokens).to(self.device)
            layer_attention_avg = 0.5 * layer_attention_avg + 0.5 * I
            
            rollout = torch.matmul(layer_attention_avg[0], rollout)
        
        # Берем attention от CLS токена
        cls_attention_rollout = rollout[0, 1:].cpu().numpy()
        
        # Нормализуем
        cls_attention_rollout = cls_attention_rollout / cls_attention_rollout.max()
        
        grid_size = int(np.sqrt(cls_attention_rollout.shape[0]))
        if grid_size * grid_size == cls_attention_rollout.shape[0]:
            attention_map = cls_attention_rollout.reshape(grid_size, grid_size)
            return self._overlay_heatmap(original_img, attention_map)
        else:
            print(f"Ошибка: невозможно преобразовать {cls_attention_rollout.shape[0]} патчей в квадратную матрицу")
            return None
    
    def compare_heads_layers(self, image, save_path=None):
        """Сравнение attention между головами и слоями с проверкой"""
        attentions, _, original_img = self.get_attention_maps(image)
        
        if attentions is None or len(attentions) == 0:
            print("Ошибка: Нет attention карт для сравнения")
            return
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Визуализируем разные слои
        layers_to_show = min(12, len(attentions))
        start_layer = max(0, len(attentions) - layers_to_show)
        
        plot_idx = 0
        for i, layer_idx in enumerate(range(start_layer, len(attentions))):
            if plot_idx >= len(axes):
                break
                
            try:
                # Усредняем по головам
                layer_attention = attentions[layer_idx][0].mean(dim=0)  # [tokens, tokens]
                
                if layer_attention.shape[0] > 1:
                    cls_attention = layer_attention[0, 1:].cpu().numpy()
                    
                    grid_size = int(np.sqrt(cls_attention.shape[0]))
                    if grid_size * grid_size == cls_attention.shape[0]:
                        attention_map = cls_attention.reshape(grid_size, grid_size)
                        
                        axes[plot_idx].imshow(attention_map, cmap='hot')
                        axes[plot_idx].set_title(f'Layer {layer_idx}')
                        axes[plot_idx].axis('off')
                        plot_idx += 1
            except Exception as e:
                print(f"Ошибка в слое {layer_idx}: {e}")
                continue
        
        # Скрываем неиспользованные subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_head_diversity(self, image):
        """Анализ разнообразия attention heads с проверкой"""
        attentions, _, _ = self.get_attention_maps(image)
        
        if attentions is None or len(attentions) == 0:
            print("Ошибка: Нет attention карт для анализа разнообразия")
            return []
        
        diversity_scores = []
        for layer_idx, layer_attention in enumerate(attentions):
            # [batch, heads, tokens, tokens]
            head_attentions = layer_attention[0]  # [heads, tokens, tokens]
            
            if head_attentions.shape[1] > 1:  # Проверяем наличие патчей
                # Считаем разнообразие между головами
                cls_attentions = head_attentions[:, 0, 1:]  # [heads, patches]
                correlations = []
                
                for i in range(self.num_heads):
                    for j in range(i+1, self.num_heads):
                        corr = F.cosine_similarity(
                            cls_attentions[i], cls_attentions[j], dim=0
                        )
                        correlations.append(corr.item())
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    diversity_scores.append((layer_idx, 1 - avg_correlation))
        
        # Визуализируем результаты
        if diversity_scores:
            layers, diversities = zip(*diversity_scores)
            plt.figure(figsize=(10, 6))
            plt.plot(layers, diversities, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Номер слоя')
            plt.ylabel('Разнообразие голов (1 - avg correlation)')
            plt.title('Разнообразие Attention Heads по слоям')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return diversity_scores
    
    def visualize_all_heads(self, image, layer_idx=0):
        """Визуализация всех голов внимания в указанном слое"""
        attentions, _, original_img = self.get_attention_maps(image)
        
        if attentions is None or len(attentions) == 0:
            print("Ошибка: Нет attention карт для визуализации")
            return
        
        if layer_idx >= len(attentions):
            print(f"Ошибка: layer_idx {layer_idx} превышает количество слоев {len(attentions)}")
            return
        
        num_heads = min(self.num_heads, 12)  # Ограничиваем количество для визуализации
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for head_idx in range(num_heads):
            if head_idx >= len(axes):
                break
                
            try:
                cls_attention = attentions[layer_idx][0, head_idx, 0, 1:].cpu().numpy()
                grid_size = int(np.sqrt(cls_attention.shape[0]))
                
                if grid_size * grid_size == cls_attention.shape[0]:
                    attention_map = cls_attention.reshape(grid_size, grid_size)
                    axes[head_idx].imshow(attention_map, cmap='hot')
                    axes[head_idx].set_title(f'Head {head_idx}')
                    axes[head_idx].axis('off')
            except Exception as e:
                print(f"Ошибка в голове {head_idx}: {e}")
                continue
        
        plt.suptitle(f'Attention Heads в слое {layer_idx}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
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
        
        heatmap = cv2.resize(attention_map, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Конвертируем RGB в BGR для OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        
        # Конвертируем обратно в RGB для matplotlib
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        ax3.imshow(superimposed_rgb)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        return fig

# Пример использования
if __name__ == "__main__":
    analyzer = AttentionAnalyzer("WinKawaks/vit-small-patch16-224")
    
    # Тестирование на случайном изображении
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Визуализация attention
    fig = analyzer.visualize_cls_attention(test_image, layer_idx=0, head_idx=0)
    
    # Attention rollout
    fig_rollout = analyzer.attention_rollout(test_image)
    
    # Сравнение слоев
    analyzer.compare_heads_layers(test_image)
    
    # Анализ разнообразия голов
    diversity = analyzer.analyze_head_diversity(test_image)
    
    # Визуализация всех голов
    analyzer.visualize_all_heads(test_image, layer_idx=0)