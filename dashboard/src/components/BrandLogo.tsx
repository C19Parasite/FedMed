import React from 'react';
import { LogoId } from '../types/theme';

interface LogoProps {
  variant?: LogoId;
  size?: number | string;
  className?: string;
  primaryColor?: string;
  secondaryColor?: string;
}

/**
 * 1. Neural Brain: Anatomical brain outline integrated with interconnected neural synaptic nodes
 */
export const NeuralBrainIcon: React.FC<Omit<LogoProps, 'variant'>> = ({
  size = 24,
  className = '',
  primaryColor = 'currentColor',
  secondaryColor = 'rgba(255, 255, 255, 0.75)',
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="FedMed Neural Brain AI"
    >
      <defs>
        <linearGradient id="neuralGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="100%" stopColor="rgba(255, 255, 255, 0.85)" />
        </linearGradient>
        <filter id="nodeGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="0.8" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Left Hemisphere Outer Sulci Contour */}
      <path
        d="M15 6C13.2 6 11.5 6.8 10.3 8.2C8.8 8.4 7.2 9.6 6.6 11.2C5.9 13.1 6.3 15.2 7.4 16.7C6.8 18.2 7.1 20 8.2 21.3C9.3 22.7 11 23.5 12.8 23.8C13.5 25.1 14.8 26 16 26"
        stroke="url(#neuralGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Right Hemisphere Outer Sulci Contour */}
      <path
        d="M17 6C18.8 6 20.5 6.8 21.7 8.2C23.2 8.4 24.8 9.6 25.4 11.2C26.1 13.1 25.7 15.2 24.6 16.7C25.2 18.2 24.9 20 23.8 21.3C22.7 22.7 21 23.5 19.2 23.8C18.5 25.1 17.2 26 16 26"
        stroke="url(#neuralGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Inter-hemispheric Longitudinal Fissure */}
      <line
        x1="16"
        y1="7"
        x2="16"
        y2="24"
        stroke="url(#neuralGrad)"
        strokeWidth="1.2"
        strokeDasharray="2 2"
        opacity="0.6"
      />

      {/* Neural Synapse Interconnect Edges */}
      <line x1="11" y1="11" x2="16" y2="13" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="21" y1="11" x2="16" y2="13" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="16" y1="13" x2="11" y2="18" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="16" y1="13" x2="21" y2="18" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="11" y1="18" x2="16" y2="21" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="21" y1="18" x2="16" y2="21" stroke="url(#neuralGrad)" strokeWidth="1" opacity="0.6" />
      <line x1="9" y1="14" x2="11" y2="11" stroke="url(#neuralGrad)" strokeWidth="0.8" opacity="0.4" />
      <line x1="23" y1="14" x2="21" y2="11" stroke="url(#neuralGrad)" strokeWidth="0.8" opacity="0.4" />

      {/* Neural Synapse Nodes */}
      <circle cx="16" cy="13" r="2" fill="#FFFFFF" filter="url(#nodeGlow)" />
      <circle cx="11" cy="11" r="1.5" fill="#FFFFFF" />
      <circle cx="21" cy="11" r="1.5" fill="#FFFFFF" />
      <circle cx="9" cy="15" r="1.3" fill="url(#neuralGrad)" opacity="0.9" />
      <circle cx="23" cy="15" r="1.3" fill="url(#neuralGrad)" opacity="0.9" />
      <circle cx="11" cy="18" r="1.5" fill="#FFFFFF" />
      <circle cx="21" cy="18" r="1.5" fill="#FFFFFF" />
      <circle cx="16" cy="21" r="1.8" fill="#FFFFFF" filter="url(#nodeGlow)" />
    </svg>
  );
};

/**
 * 2. Brain + Circuit: Anatomical cerebral architecture combined with orthogonal PCB circuit traces and contact vias
 */
export const BrainCircuitIcon: React.FC<Omit<LogoProps, 'variant'>> = ({
  size = 24,
  className = '',
  primaryColor = 'currentColor',
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="FedMed Brain Circuit AI"
    >
      <defs>
        <linearGradient id="circuitGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="100%" stopColor="rgba(255, 255, 255, 0.88)" />
        </linearGradient>
      </defs>

      {/* Left Hemisphere Structural Outline */}
      <path
        d="M15 6C12 6 9.5 7.5 8 10C6.5 12.5 6.5 15.5 8 18C7 19.5 7.5 21.5 9 23C10.5 24.5 12.5 25.5 15 25.8"
        stroke="url(#circuitGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
      />

      {/* Right Hemisphere Structural Outline */}
      <path
        d="M17 6C20 6 22.5 7.5 24 10C25.5 12.5 25.5 15.5 24 18C25 19.5 24.5 21.5 23 23C21.5 24.5 19.5 25.5 17 25.8"
        stroke="url(#circuitGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
      />

      {/* Central Microprocessor Bus */}
      <rect
        x="13.5"
        y="12"
        width="5"
        height="7"
        rx="1.5"
        stroke="url(#circuitGrad)"
        strokeWidth="1.2"
        fill="rgba(255, 255, 255, 0.15)"
      />

      {/* Orthogonal PCB Bus Lines with Terminal Pads */}
      <path d="M13.5 14H10L8.5 12.5" stroke="url(#circuitGrad)" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="8" cy="12" r="1.2" fill="#FFFFFF" />

      <path d="M18.5 14H22L23.5 12.5" stroke="url(#circuitGrad)" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="24" cy="12" r="1.2" fill="#FFFFFF" />

      <path d="M13.5 17H10L8.5 18.5" stroke="url(#circuitGrad)" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="8" cy="19" r="1.2" fill="#FFFFFF" />

      <path d="M18.5 17H22L23.5 18.5" stroke="url(#circuitGrad)" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="24" cy="19" r="1.2" fill="#FFFFFF" />

      {/* Vertical Data Bus */}
      <line x1="16" y1="7" x2="16" y2="12" stroke="url(#circuitGrad)" strokeWidth="1.2" />
      <circle cx="16" cy="7" r="1.2" fill="#FFFFFF" />

      <line x1="16" y1="19" x2="16" y2="24" stroke="url(#circuitGrad)" strokeWidth="1.2" />
      <circle cx="16" cy="24" r="1.2" fill="#FFFFFF" />
    </svg>
  );
};

/**
 * 3. Medical Intelligence: Geometric brain contour enclosing a clinical cross + vital signal pulse
 */
export const MedicalIntelligenceIcon: React.FC<Omit<LogoProps, 'variant'>> = ({
  size = 24,
  className = '',
  primaryColor = 'currentColor',
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="FedMed Medical Intelligence"
    >
      <defs>
        <linearGradient id="medGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="100%" stopColor="rgba(255, 255, 255, 0.88)" />
        </linearGradient>
      </defs>

      {/* Symmetrical Geometric Cerebral Silhouette */}
      <path
        d="M16 6C12.5 6 9.5 7.5 8 10.5C6.5 13.5 6.8 17 8.5 19.8C8 21.2 8.5 22.8 9.8 23.8C11.5 25.2 13.8 25.8 16 25.8C18.2 25.8 20.5 25.2 22.2 23.8C23.5 22.8 24 21.2 23.5 19.8C25.2 17 25.5 13.5 24 10.5C22.5 7.5 19.5 6 16 6Z"
        stroke="url(#medGrad)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Medical Cross (Geometric Precision) */}
      <path
        d="M14.5 10H17.5V13.5H21V16.5H17.5V20H14.5V16.5H11V13.5H14.5V10Z"
        fill="rgba(255, 255, 255, 0.22)"
        stroke="url(#medGrad)"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />

      {/* Vital Diagnostic Pulse / Cardiogram Trace through Cross Center */}
      <path
        d="M9 15H13L14.5 12L16 18L17.5 13.5L19 15H23"
        stroke="#FFFFFF"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

/**
 * Universal BrandLogo Component
 */
export const BrandLogo: React.FC<LogoProps> = ({
  variant = 'neural-brain',
  size = 24,
  className = '',
  primaryColor,
  secondaryColor,
}) => {
  switch (variant) {
    case 'brain-circuit':
      return <BrainCircuitIcon size={size} className={className} primaryColor={primaryColor} />;
    case 'medical-intelligence':
      return <MedicalIntelligenceIcon size={size} className={className} primaryColor={primaryColor} />;
    case 'neural-brain':
    default:
      return (
        <NeuralBrainIcon
          size={size}
          className={className}
          primaryColor={primaryColor}
          secondaryColor={secondaryColor}
        />
      );
  }
};
