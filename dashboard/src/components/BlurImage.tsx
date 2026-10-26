import React, { useState } from 'react';

interface BlurImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  blurPlaceholder: string;
  alt: string;
  containerClassName?: string;
}

export const BlurImage: React.FC<BlurImageProps> = ({
  src,
  blurPlaceholder,
  alt,
  className = '',
  containerClassName = '',
  ...props
}) => {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    <div className={`relative overflow-hidden ${containerClassName}`}>
      {/* Blurred Micro-placeholder displayed immediately */}
      <div
        className={`absolute inset-0 bg-cover bg-center transition-opacity duration-500 scale-110 filter blur-md ${
          isLoaded ? 'opacity-0' : 'opacity-100'
        }`}
        style={{ backgroundImage: `url(${blurPlaceholder})` }}
        aria-hidden="true"
      />

      {/* Main Full-Resolution Image fading in on load */}
      {src && (
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          className={`transition-opacity duration-500 relative z-10 ${
            isLoaded ? 'opacity-100' : 'opacity-0'
          } ${className}`}
          {...props}
        />
      )}
    </div>
  );
};
