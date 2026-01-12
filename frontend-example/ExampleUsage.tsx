// Example usage in your ecommerce product page
import React from 'react';
import { ProductColorPicker } from './ProductColorPicker';
import { recolorClothing } from './recolorApi';

// ============================================
// OPTION 1: Using the ProductColorPicker component
// ============================================

interface ProductPageProps {
  product: {
    id: string;
    name: string;
    imageUrl: string;
    price: number;
  };
}

export const ProductPage: React.FC<ProductPageProps> = ({ product }) => {
  const [displayImage, setDisplayImage] = React.useState(product.imageUrl);

  const handleImageChange = (newImageUrl: string) => {
    setDisplayImage(newImageUrl);
    // Optionally update cart preview, etc.
  };

  return (
    <div className="product-page">
      <h1>{product.name}</h1>
      
      <ProductColorPicker
        originalImageUrl={product.imageUrl}
        onImageChange={handleImageChange}
        onError={(err) => console.error('Recolor error:', err)}
      />
      
      <p className="price">${product.price}</p>
      <button className="add-to-cart">Add to Cart</button>
    </div>
  );
};


// ============================================
// OPTION 2: Direct API usage (for custom integration)
// ============================================

export const useClothingRecolor = () => {
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  const recolor = React.useCallback(async (
    imageUrl: string,
    hexColor: string
  ): Promise<string | null> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await recolorClothing(imageUrl, hexColor);
      return result;
    } catch (err) {
      setError(err as Error);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { recolor, isLoading, error };
};

// Usage in component:
// const { recolor, isLoading } = useClothingRecolor();
// const newImage = await recolor(originalUrl, '#FF0000');


// ============================================
// OPTION 3: Integration with existing color picker
// ============================================

interface ExistingColorPickerIntegrationProps {
  imageUrl: string;
  selectedColor: string; // hex color from your existing picker
  onRecoloredImage: (url: string) => void;
}

export const ExistingColorPickerIntegration: React.FC<ExistingColorPickerIntegrationProps> = ({
  imageUrl,
  selectedColor,
  onRecoloredImage,
}) => {
  React.useEffect(() => {
    if (!selectedColor) return;
    
    let cancelled = false;
    
    recolorClothing(imageUrl, selectedColor)
      .then((newUrl) => {
        if (!cancelled) {
          onRecoloredImage(newUrl);
        }
      })
      .catch(console.error);
    
    return () => { cancelled = true; };
  }, [imageUrl, selectedColor, onRecoloredImage]);

  return null; // This is just a logic component
};
