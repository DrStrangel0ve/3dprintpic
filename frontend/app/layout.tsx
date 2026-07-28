import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Script from 'next/script';

import './globals.css';

const inter = Inter({ subsets: ['latin'] });
const voiceflowProjectId = process.env.NEXT_PUBLIC_VOICEFLOW_PROJECT_ID;

export const metadata: Metadata = {
  title: '3D Print a Picture',
  description: 'Turn a photo or video into a printable STL.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
        {voiceflowProjectId && (
          <Script
            id="voiceflow-widget"
            strategy="lazyOnload"
            dangerouslySetInnerHTML={{
              __html: `
                (function(d, t) {
                  var v = d.createElement(t), s = d.getElementsByTagName(t)[0];
                  v.onload = function() {
                    window.voiceflow.chat.load({
                      verify: { projectID: ${JSON.stringify(voiceflowProjectId)} },
                      url: 'https://general-runtime.voiceflow.com'
                    });
                  };
                  v.src = 'https://cdn.voiceflow.com/widget/bundle.mjs';
                  v.type = 'text/javascript';
                  s.parentNode.insertBefore(v, s);
                })(document, 'script');
              `,
            }}
          />
        )}
      </body>
    </html>
  );
}
