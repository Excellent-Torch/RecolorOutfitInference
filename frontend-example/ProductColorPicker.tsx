import React, { useState, useCallback } from 'react';
import { recolorClothing, DEFAULT_COLORS, ColorOption } from './recolorApi';

interface ProductColorPickerProps {
  /** Original product image URL */
  originalImageUrl: string;
  /** Available color options */
  colors?: ColorOption[];
  /** Callback when image is successfully recolored */
  onImageChange?: (newImageUrl: string) => void;
  /** Callback for errors */
  onError?: (error: Error) => void;
}

export const ProductColorPicker: React.FC<ProductColorPickerProps> = ({
  originalImageUrl,
  colors = DEFAULT_COLORS,
  onImageChange,
  onError,
}) => {
  const [currentImage, setCurrentImage] = useState(originalImageUrl);
  const [selectedColor, setSelectedColor] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleColorSelect = useCallback(async (color: ColorOption) => {
    if (isLoading) return;
    
    setIsLoading(true);
    setSelectedColor(color.hex);
    
    try {
      const recoloredUrl = await recolorClothing(originalImageUrl, color.hex);
      setCurrentImage(recoloredUrl);
      onImageChange?.(recoloredUrl);
    } catch (error) {
      console.error('Recolor failed:', error);
      onError?.(error as Error);
      // Revert to original on error
      setCurrentImage(originalImageUrl);
    } finally {
      setIsLoading(false);
    }
  }, [originalImageUrl, isLoading, onImageChange, onError]);

  const handleReset = useCallback(() => {
    setCurrentImage(originalImageUrl);
    setSelectedColor(null);
    onImageChange?.(originalImageUrl);
  }, [originalImageUrl, onImageChange]);

  return (
    <div className="product-color-picker">
      {/* Product Image */}
      <div className="relative">
        <img 
          src={currentImage} 
          alt="Product" 
          className={`product-image ${isLoading ? 'opacity-50' : ''}`}
        />
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="spinner" />
          </div>
        )}
      </div>

      {/* Color Options */}
      <div className="color-options mt-4">
        <p className="text-sm font-medium mb-2">Select Color:</p>
        <div className="flex flex-wrap gap-2">
          {colors.map((color) => (
            <button
              key={color.hex}
              onClick={() => handleColorSelect(color)}
              disabled={isLoading}
              className={`
                w-8 h-8 rounded-full border-2 transition-all
                ${selectedColor === color.hex ? 'ring-2 ring-offset-2 ring-blue-500' : ''}
                ${isLoading ? 'cursor-not-allowed' : 'cursor-pointer hover:scale-110'}
              `}
              style={{ backgroundColor: color.hex }}
              title={color.name}
              aria-label={`Select ${color.name}`}
            />
          ))}
        </div>
      </div>

      {/* Reset Button */}
      {selectedColor && (
        <button
          onClick={handleReset}
          className="mt-3 text-sm text-gray-600 hover:text-gray-800 underline"
        >
          Reset to original
        </button>
      )}
    </div>
  );
};

export default ProductColorPicker;
