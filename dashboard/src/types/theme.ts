export type ThemeId =
  | 'clinical-violet'
  | 'medical-teal'
  | 'midnight-sapphire'
  | 'graphite-cyan'
  | 'emerald-healthcare'
  | 'arctic-blue';

export type LogoId = 'neural-brain' | 'brain-circuit' | 'medical-intelligence';

export interface ThemeDefinition {
  id: ThemeId;
  name: string;
  tagline: string;
  styleDescription: string;
  isDark: boolean;
  colors: {
    primary: string;
    secondary: string;
    background: string;
    shell: string;
    card: string;
    surfaceSoft: string;
    cardBorder: string;
    text: string;
    textMuted: string;
    accent: string;
    sidebarBg: string;
  };
  palette: {
    p50: string;
    p100: string;
    p200: string;
    p300: string;
    p500: string;
    p600: string;
    p700: string;
    p800: string;
    p900: string;
    p950: string;
  };
  heroGradient: string;
}

export interface LogoDefinition {
  id: LogoId;
  name: string;
  subtitle: string;
  description: string;
}

export const THEMES: Record<ThemeId, ThemeDefinition> = {
  'clinical-violet': {
    id: 'clinical-violet',
    name: 'Theme A — Clinical Violet',
    tagline: 'Modern medical AI, soft lavender surfaces, clean and professional',
    styleDescription: 'Balanced clinical elegance with lavender depth and clean whites.',
    isDark: false,
    colors: {
      primary: '#7857FF',
      secondary: '#A78BFA',
      background: '#F4F2FA',
      shell: '#E9E7F2',
      card: '#FFFFFF',
      surfaceSoft: '#F8F7FC',
      cardBorder: '#EBE7F5',
      text: '#191827',
      textMuted: '#71717A',
      accent: '#22C55E',
      sidebarBg: '#111116',
    },
    palette: {
      p50: '#F8F6FE',
      p100: '#F1EDFF',
      p200: '#E4DCFE',
      p300: '#D9CFFB',
      p500: '#9475F7',
      p600: '#7857FF',
      p700: '#6441E6',
      p800: '#4D2DC0',
      p900: '#341A8E',
      p950: '#1F0D59',
    },
    heroGradient: 'linear-gradient(135deg, #DDD2FD 0%, #EFEAFF 50%, #F6F3FF 100%)',
  },
  'medical-teal': {
    id: 'medical-teal',
    name: 'Theme B — Medical Teal',
    tagline: 'Clinical, calm, trustworthy, healthcare-oriented',
    styleDescription: 'High-trust hospital informatics with soothing teal and seafoam tones.',
    isDark: false,
    colors: {
      primary: '#0D9488',
      secondary: '#5EEAD4',
      background: '#F0FDFA',
      shell: '#E0F7F4',
      card: '#FFFFFF',
      surfaceSoft: '#F4FBF9',
      cardBorder: '#CCFBF1',
      text: '#134E4A',
      textMuted: '#4F7571',
      accent: '#22C55E',
      sidebarBg: '#0F1F1D',
    },
    palette: {
      p50: '#F0FDFA',
      p100: '#CCFBF1',
      p200: '#99F6E4',
      p300: '#5EEAD4',
      p500: '#14B8A6',
      p600: '#0D9488',
      p700: '#0F766E',
      p800: '#115E59',
      p900: '#134E4A',
      p950: '#042F2E',
    },
    heroGradient: 'linear-gradient(135deg, #CCFBF1 0%, #E6FAF6 50%, #F0FDFA 100%)',
  },
  'midnight-sapphire': {
    id: 'midnight-sapphire',
    name: 'Theme C — Midnight Sapphire',
    tagline: 'Premium AI research platform with dark navy interface and bright blue highlights',
    styleDescription: 'Deep oceanic dark mode with luminous cobalt signals and high clarity.',
    isDark: true,
    colors: {
      primary: '#3B82F6',
      secondary: '#60A5FA',
      background: '#0B1220',
      shell: '#060A12',
      card: '#111C30',
      surfaceSoft: '#16233B',
      cardBorder: '#1E2E4A',
      text: '#E5EEFF',
      textMuted: '#94A3B8',
      accent: '#34D399',
      sidebarBg: '#080D18',
    },
    palette: {
      p50: '#16233B',
      p100: '#1D3256',
      p200: '#2563EB33',
      p300: '#3B82F64D',
      p500: '#60A5FA',
      p600: '#3B82F6',
      p700: '#2563EB',
      p800: '#1D4ED8',
      p900: '#1E3A8A',
      p950: '#172554',
    },
    heroGradient: 'linear-gradient(135deg, #162B4D 0%, #111C30 50%, #0B1220 100%)',
  },
  'graphite-cyan': {
    id: 'graphite-cyan',
    name: 'Theme D — Graphite & Electric Cyan',
    tagline: 'Technical, high-contrast, futuristic AI infrastructure dashboard',
    styleDescription: 'Cybernetic dark slate with razor-sharp electric cyan luminescence.',
    isDark: true,
    colors: {
      primary: '#06B6D4',
      secondary: '#67E8F9',
      background: '#111418',
      shell: '#0A0C0E',
      card: '#1B222A',
      surfaceSoft: '#222B35',
      cardBorder: '#283542',
      text: '#F1F5F9',
      textMuted: '#94A3B8',
      accent: '#4ADE80',
      sidebarBg: '#0C0E11',
    },
    palette: {
      p50: '#16252C',
      p100: '#1B333E',
      p200: '#0891B233',
      p300: '#06B6D44D',
      p500: '#22D3EE',
      p600: '#06B6D4',
      p700: '#0891B2',
      p800: '#0E7490',
      p900: '#155E75',
      p950: '#083344',
    },
    heroGradient: 'linear-gradient(135deg, #1E313D 0%, #1B222A 50%, #111418 100%)',
  },
  'emerald-healthcare': {
    id: 'emerald-healthcare',
    name: 'Theme E — Emerald Healthcare',
    tagline: 'Natural, reassuring, healthcare-focused, with restrained green accents',
    styleDescription: 'Vital organic green theme fostering calm confidence and diagnostic precision.',
    isDark: false,
    colors: {
      primary: '#059669',
      secondary: '#6EE7B7',
      background: '#F0FDF4',
      shell: '#DCF5E5',
      card: '#FFFFFF',
      surfaceSoft: '#F4FBF6',
      cardBorder: '#D1FAE5',
      text: '#16352B',
      textMuted: '#4D6D62',
      accent: '#6366F1',
      sidebarBg: '#0F221B',
    },
    palette: {
      p50: '#F0FDF4',
      p100: '#D1FAE5',
      p200: '#A7F3D0',
      p300: '#6EE7B7',
      p500: '#10B981',
      p600: '#059669',
      p700: '#047857',
      p800: '#065F46',
      p900: '#064E3B',
      p950: '#022C22',
    },
    heroGradient: 'linear-gradient(135deg, #D1FAE5 0%, #E8FAF0 50%, #F0FDF4 100%)',
  },
  'arctic-blue': {
    id: 'arctic-blue',
    name: 'Theme F — Arctic Blue',
    tagline: 'Clean hospital technology platform with crisp blue accents and warm highlights',
    styleDescription: 'Crisp medical workstation aesthetic with ultramarine focus and warm amber accents.',
    isDark: false,
    colors: {
      primary: '#2563EB',
      secondary: '#93C5FD',
      background: '#F1F7FE',
      shell: '#E0EDFC',
      card: '#FFFFFF',
      surfaceSoft: '#F6F9FE',
      cardBorder: '#DBEAFE',
      text: '#172B4D',
      textMuted: '#5C6F8E',
      accent: '#F59E0B',
      sidebarBg: '#0E1B2E',
    },
    palette: {
      p50: '#EFF6FF',
      p100: '#DBEAFE',
      p200: '#BFDBFE',
      p300: '#93C5FD',
      p500: '#3B82F6',
      p600: '#2563EB',
      p700: '#1D4ED8',
      p800: '#1E40AF',
      p900: '#1E3A8A',
      p950: '#172554',
    },
    heroGradient: 'linear-gradient(135deg, #DBEAFE 0%, #EDF4FE 50%, #F1F7FE 100%)',
  },
};

export const LOGO_VARIANTS: Record<LogoId, LogoDefinition> = {
  'neural-brain': {
    id: 'neural-brain',
    name: 'Neural Brain',
    subtitle: 'Connected Synaptic Topology',
    description: 'Anatomical dual-hemisphere contours interwoven with neural network nodes and synaptic connection pathways.',
  },
  'brain-circuit': {
    id: 'brain-circuit',
    name: 'Brain + Circuit',
    subtitle: 'Silicon Bio-Informatics',
    description: 'Crisp cerebral architecture integrated with micro-bus PCB traces, terminal via pads, and hardware-accelerated ML lines.',
  },
  'medical-intelligence': {
    id: 'medical-intelligence',
    name: 'Medical Intelligence',
    subtitle: 'Clinical Cross & Vital Signals',
    description: 'Geometric cerebral silhouette housing a clean clinical cross and cardiogram rhythm motif, representing diagnostic intelligence.',
  },
};
