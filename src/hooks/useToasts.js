import { useCallback, useState } from 'react';

export function useToasts() {
  const [toasts, setToasts] = useState([]);

  const pushToast = useCallback((message, type = 'info', durationMs) => {
    const id = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
    setToasts((prev) => [...prev, { id, message, type, durationMs }]);
    return id;
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return { toasts, pushToast, dismissToast };
}
