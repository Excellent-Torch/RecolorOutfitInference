// Clothing Recolor API Client for React/TypeScript
// Add this to your ecommerce platform

const API_URL = process.env.REACT_APP_RECOLOR_API_URL || 'http://localhost:8000';

/**
 * Recolor clothing in an image
 * @param imageSource - Image URL, File, or Blob
 * @param hexColor - Target color in hex format (#RRGGBB)
 * @returns Promise<string> - Object URL of the recolored image
 */
export async function recolorClothing(
  imageSource: string | File | Blob,
  hexColor: string
): Promise<string> {
  const formData = new FormData();
  
  // Handle different image sources
  if (typeof imageSource === 'string') {
    // If it's a URL, fetch the image first
    const response = await fetch(imageSource);
    const blob = await response.blob();
    formData.append('image', blob, 'image.jpg');
  } else {
    formData.append('image', imageSource);
  }
  
  formData.append('color', hexColor);
  
  const response = await fetch(`${API_URL}/recolor`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Recolor failed: ${error}`);
  }
  
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

/**
 * Check if the recolor API is available
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

// Types for your color picker component
export interface ColorOption {
  name: string;
  hex: string;
}

export const DEFAULT_COLORS: ColorOption[] = [
  { name: 'Red', hex: '#FF0000' },
  { name: 'Blue', hex: '#0066CC' },
  { name: 'Green', hex: '#228B22' },
  { name: 'Yellow', hex: '#FFD700' },
  { name: 'Purple', hex: '#8B008B' },
  { name: 'Pink', hex: '#FF69B4' },
  { name: 'Orange', hex: '#FF8C00' },
  { name: 'Black', hex: '#1A1A1A' },
  { name: 'White', hex: '#F5F5F5' },
  { name: 'Navy', hex: '#000080' },
  { name: 'Brown', hex: '#8B4513' },
  { name: 'Cyan', hex: '#00CED1' },
];
