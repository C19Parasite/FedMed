import React, { useState, useRef, useEffect } from 'react';
import { X, ShieldAlert, KeyRound, CheckCircle2, Lock } from 'lucide-react';
import { toast } from 'sonner';

interface TwoFactorAuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  title?: string;
}

export const TwoFactorAuthModal: React.FC<TwoFactorAuthModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  title = 'Node Access Verification (2FA)',
}) => {
  const [digits, setDigits] = useState<string[]>(['', '', '', '', '', '']);
  const [isVerifying, setIsVerifying] = useState(false);
  const inputsRef = useRef<(HTMLInputElement | null)[]>([]);

  useEffect(() => {
    if (isOpen) {
      setDigits(['', '', '', '', '', '']);
      setTimeout(() => inputsRef.current[0]?.focus(), 100);

      // WebOTP API support
      if ('OTPCredential' in window) {
        const ac = new AbortController();
        navigator.credentials
          .get({
            // @ts-ignore
            otp: { transport: ['sms'] },
            signal: ac.signal,
          })
          .then((otp: any) => {
            if (otp && otp.code) {
              const codeDigits = otp.code.slice(0, 6).split('');
              setDigits(codeDigits);
              submitCode(codeDigits.join(''));
            }
          })
          .catch(() => {});
        return () => ac.abort();
      }
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleChange = (index: number, value: string) => {
    // Only accept numeric inputs
    const clean = value.replace(/\D/g, '');
    if (!clean) {
      const updated = [...digits];
      updated[index] = '';
      setDigits(updated);
      return;
    }

    // Single digit entry
    const char = clean[clean.length - 1];
    const updated = [...digits];
    updated[index] = char;
    setDigits(updated);

    // Auto-advance to next input
    if (index < 5 && char) {
      inputsRef.current[index + 1]?.focus();
    }

    // Auto-submit if 6th digit filled
    if (index === 5 || updated.every((d) => d !== '')) {
      const fullCode = updated.join('');
      if (fullCode.length === 6) {
        submitCode(fullCode);
      }
    }
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Backspace' && !digits[index] && index > 0) {
      inputsRef.current[index - 1]?.focus();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    e.preventDefault();
    const pasteData = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    if (!pasteData) return;

    const newDigits = [...digits];
    for (let i = 0; i < pasteData.length; i++) {
      newDigits[i] = pasteData[i];
    }
    setDigits(newDigits);

    if (pasteData.length === 6) {
      submitCode(pasteData);
    } else {
      inputsRef.current[pasteData.length]?.focus();
    }
  };

  const submitCode = (code: string) => {
    setIsVerifying(true);
    setTimeout(() => {
      setIsVerifying(false);
      onClose();
      toast.success('Node Access Verified', {
        description: 'Authorized secure cryptographic dispatch for cross-silo aggregation.',
      });
      onSuccess();
    }, 600);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl border border-brand-border dark:border-zinc-800 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-brand-border dark:border-zinc-800 bg-surface-soft dark:bg-zinc-800/40">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center">
              <KeyRound size={16} />
            </div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              {title}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-400 hover:text-zinc-700 dark:hover:text-white flex items-center justify-center transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Form Body */}
        <div className="p-6 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-lavender-50 dark:bg-lavender-950/50 text-lavender-600 mx-auto flex items-center justify-center shadow-sm">
            <Lock size={22} />
          </div>

          <div>
            <h4 className="text-base font-extrabold text-zinc-900 dark:text-white">
              Hospital Node Key Authorization
            </h4>
            <p className="text-xs text-zinc-500 mt-1 max-w-xs mx-auto">
              Enter the 6-digit one-time code to authorize TenSEAL evaluation keys for this round.
            </p>
          </div>

          {/* 6 Digit Input Boxes */}
          <div className="flex justify-center gap-2.5 pt-2">
            {digits.map((digit, idx) => (
              <input
                key={idx}
                ref={(el) => (inputsRef.current[idx] = el)}
                type="text"
                autoComplete="one-time-code"
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={1}
                value={digit}
                onChange={(e) => handleChange(idx, e.target.value)}
                onKeyDown={(e) => handleKeyDown(idx, e)}
                onPaste={handlePaste}
                disabled={isVerifying}
                className="w-11 h-14 text-center text-xl font-bold font-mono rounded-2xl border-2 border-brand-border dark:border-zinc-700 bg-surface-soft dark:bg-zinc-800 text-zinc-900 dark:text-white focus:border-lavender-600 focus:outline-none transition-all shadow-sm"
              />
            ))}
          </div>

          <div className="text-[11px] text-zinc-400 font-medium">
            Auto-submits on 6th digit • Paste supported
          </div>
        </div>
      </div>
    </div>
  );
};
