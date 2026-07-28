import type { HTMLAttributes, ReactNode } from 'react';
import { Box, ImageUp, ScanLine, type LucideIcon } from 'lucide-react';

import { cn } from '@/lib/utils';

type WorkspaceShellProps = {
  children: ReactNode;
  className?: string;
};

type WorkspaceColumnProps = HTMLAttributes<HTMLElement> & {
  children: ReactNode;
  label: string;
  name: 'input' | 'geometry' | 'output';
};

const columnMeta: Record<
  WorkspaceColumnProps['name'],
  { index: string; title: string; icon: LucideIcon; tone: string }
> = {
  input: { index: '01', title: 'Source', icon: ImageUp, tone: 'text-blue-700' },
  geometry: { index: '02', title: 'Geometry', icon: ScanLine, tone: 'text-orange-700' },
  output: { index: '03', title: 'Print', icon: Box, tone: 'text-emerald-700' },
};

export function WorkspaceShell({ children, className }: WorkspaceShellProps) {
  return (
    <div
      className={cn(
        'workspace-shell mx-auto grid w-full max-w-[1536px] grid-cols-1 gap-3 px-3 py-3 sm:px-4 sm:py-4',
        'md:grid-cols-[minmax(280px,360px)_minmax(0,1fr)]',
        'xl:h-dvh xl:grid-cols-[minmax(300px,420px)_minmax(380px,1fr)_minmax(310px,390px)] xl:overflow-hidden',
        className,
      )}
      data-testid="workspace-shell"
    >
      {children}
    </div>
  );
}

export function WorkspaceColumn({ children, className, label, name, ...props }: WorkspaceColumnProps) {
  const meta = columnMeta[name];
  const Icon = meta.icon;
  const placement = {
    input: 'md:col-start-1 md:row-start-1 xl:col-auto xl:row-auto',
    geometry: 'md:col-start-2 md:row-span-2 md:row-start-1 xl:col-auto xl:row-auto xl:row-span-1',
    output: 'md:col-start-1 md:row-start-2 xl:col-auto xl:row-auto',
  }[name];

  return (
    <section
      {...props}
      aria-label={label}
      className={cn(
        'workspace-column flex min-h-0 min-w-0 flex-col gap-3',
        placement,
        'xl:overflow-y-auto xl:overscroll-contain xl:pr-1 xl:[scrollbar-gutter:stable]',
        className,
      )}
      data-testid={`workspace-${name}`}
      data-workspace-column={name}
    >
      <div className="workspace-stage-rail">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className={cn('grid h-7 w-7 shrink-0 place-items-center', meta.tone)} aria-hidden="true">
            <Icon className="h-4 w-4" strokeWidth={1.8} />
          </span>
          <span className="truncate text-sm font-semibold text-zinc-900">{meta.title}</span>
        </div>
        <span className="font-mono text-[11px] font-semibold text-zinc-400">{meta.index}</span>
      </div>
      {children}
    </section>
  );
}