'use client';

import { useEffect } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';

export default function WorkspaceError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error('Workspace render failed', error);
  }, [error]);

  return (
    <main className="grid min-h-dvh place-items-center bg-zinc-50 px-4 text-zinc-950">
      <section className="w-full max-w-lg border border-red-200 bg-white p-6 shadow-sm" role="alert">
        <AlertTriangle className="h-8 w-8 text-red-700" />
        <h1 className="mt-4 text-xl font-semibold">The workspace could not render</h1>
        <p className="mt-2 text-sm leading-6 text-zinc-600">
          Your imported file is still on this device. Retry the workspace; if the problem remains, refresh the page.
        </p>
        {error.digest && <p className="mt-3 break-all text-xs text-zinc-400">Error reference: {error.digest}</p>}
        <Button className="mt-5 gap-2" onClick={reset}>
          <RotateCcw className="h-4 w-4" />
          Retry workspace
        </Button>
      </section>
    </main>
  );
}
