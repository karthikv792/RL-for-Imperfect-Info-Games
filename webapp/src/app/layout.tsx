import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sequence AI - Challenge the Machine",
  description: "Play Sequence against state-of-the-art AI agents",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
