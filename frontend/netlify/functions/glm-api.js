const HOP_BY_HOP_HEADERS = new Set([
  'connection', 'content-encoding', 'content-length', 'host', 'keep-alive',
  'proxy-authenticate', 'proxy-authorization', 'te', 'trailer',
  'transfer-encoding', 'upgrade',
]);

function json(statusCode, detail) {
  return {
    statusCode,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
    body: JSON.stringify({ detail }),
  };
}

function apiOrigin() {
  const configured = process.env.GLM_API_ORIGIN?.trim();
  if (!configured) return null;
  try {
    const parsed = new URL(configured);
    if (!['http:', 'https:'].includes(parsed.protocol)) return null;
    return parsed.toString().replace(/\/+$/, '');
  } catch {
    return null;
  }
}

function filteredRequestHeaders(headers = {}) {
  const output = {};
  for (const [name, value] of Object.entries(headers)) {
    if (value && !HOP_BY_HOP_HEADERS.has(name.toLowerCase())) output[name] = value;
  }
  return output;
}

function filteredResponseHeaders(headers) {
  const output = {};
  headers.forEach((value, name) => {
    if (!HOP_BY_HOP_HEADERS.has(name.toLowerCase())) output[name] = value;
  });
  output['cache-control'] = output['cache-control'] ?? 'no-store';
  return output;
}

function targetFor(event, origin) {
  const params = new URLSearchParams(event.rawQuery ?? '');
  const proxyPath = params.get('proxy_path') ?? event.queryStringParameters?.proxy_path;
  params.delete('proxy_path');
  const targetPath = proxyPath || event.path?.replace(/^\/api(?=\/|$)/, '') || '/';
  if (!targetPath.startsWith('/') || targetPath.includes('\\') || targetPath.includes('\r') || targetPath.includes('\n')) {
    throw new Error('Invalid proxy path.');
  }
  const query = params.toString();
  return `${origin}${targetPath}${query ? `?${query}` : ''}`;
}

export const handler = async (event) => {
  try {
    const origin = apiOrigin();
    if (!origin) return json(503, 'Netlify proxy is not configured. Set GLM_API_ORIGIN to the full Render API URL.');

    const target = targetFor(event, origin);
    const method = (event.httpMethod ?? 'GET').toUpperCase();
    const body = method === 'GET' || method === 'HEAD' || !event.body
      ? undefined
      : Buffer.from(event.body, event.isBase64Encoded ? 'base64' : 'utf8');

    const upstream = await fetch(target, {
      method,
      headers: filteredRequestHeaders(event.headers),
      body,
      redirect: 'manual',
      signal: AbortSignal.timeout(120_000),
    });
    const payload = Buffer.from(await upstream.arrayBuffer());
    const headers = filteredResponseHeaders(upstream.headers);
    if (upstream.status >= 500) {
      headers['x-glm-upstream-status'] = String(upstream.status);
    }
    return {
      statusCode: upstream.status,
      headers,
      body: payload.toString('base64'),
      isBase64Encoded: true,
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'Unknown proxy failure';
    return json(502, `Netlify could not reach the Render API: ${detail}`);
  }
};
