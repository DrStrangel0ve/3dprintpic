/** @type {import('next').NextConfig} */
const cliPort = (() => {
  for (let index = 0; index < process.argv.length; index += 1) {
    const argument = process.argv[index];
    if (argument === '-p' || argument === '--port') {
      return process.argv[index + 1];
    }
    if (argument.startsWith('--port=')) {
      return argument.slice('--port='.length);
    }
  }
  return undefined;
})();

const isDevelopment =
  process.env.NODE_ENV === 'development' || process.argv.includes('dev');
const developmentPort = process.env.PORT || cliPort || '3000';

const nextConfig = {
  distDir:
    process.env.NEXT_DIST_DIR ||
    (isDevelopment ? `.next-dev-${developmentPort}` : '.next'),
};

export default nextConfig;
