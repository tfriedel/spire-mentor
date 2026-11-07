/**
 * Fetch the engine wasm for the wasm-bindgen glue's `init`.
 *
 * The static build ships the engine GZIPPED (`*.wasm.gz`): built with SIMD it
 * is over Cloudflare Pages' 25 MiB per-file cap, gzipped it is ~7 MB
 * (`vite.config.ts` points the worker's `?url` import at the .gz,
 * `scripts/workbench_static.py` writes it and gates dist on the cap). The host
 * serves it as opaque bytes (`_headers`: octet-stream, `no-transform`), so the
 * worker inflates it itself with `DecompressionStream` and hands the glue a
 * `Response` typed `application/wasm`, which keeps
 * `WebAssembly.instantiateStreaming` (compile while the bytes arrive).
 *
 * If a host ever does add `Content-Encoding: gzip`, the browser has already
 * inflated the body; the gzip magic is sniffed, so that case still loads.
 * The dev server serves the plain `.wasm`, which passes straight through.
 */

const GZIP_MAGIC = [0x1f, 0x8b];

export const NO_DECOMPRESSION =
  "This browser cannot unpack the workbench engine (no DecompressionStream). " +
  "It needs Chrome 80+, Edge 80+, Firefox 113+ or Safari 16.4+.";

/** The engine at `url` as a `Response` the glue's `init` can stream-compile. */
export async function engineResponse(url: string, fetchFn: typeof fetch = fetch): Promise<Response> {
  const res = await fetchFn(url);
  if (!res.ok) throw new Error(`engine download failed: ${url}: ${res.status} ${res.statusText}`);
  if (!new URL(url, "http://x/").pathname.endsWith(".gz")) return res;
  if (typeof DecompressionStream === "undefined") throw new Error(NO_DECOMPRESSION);
  if (!res.body) throw new Error(`engine download failed: ${url}: empty body`);
  const { head, stream } = await peek(res.body, GZIP_MAGIC.length);
  const gzipped = GZIP_MAGIC.every((b, i) => head[i] === b);
  const body = gzipped ? stream.pipeThrough(new DecompressionStream("gzip")) : stream;
  return new Response(body, { headers: { "content-type": "application/wasm" } });
}

/** The first `n` bytes of `body`, and a stream that still yields all of it. */
async function peek(
  body: ReadableStream<Uint8Array>,
  n: number,
): Promise<{ head: Uint8Array; stream: ReadableStream<Uint8Array> }> {
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let have = 0;
  while (have < n) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    have += value.length;
  }
  const head = new Uint8Array(Math.min(n, have));
  let at = 0;
  for (const c of chunks) {
    if (at >= head.length) break;
    const take = c.subarray(0, head.length - at);
    head.set(take, at);
    at += take.length;
  }
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c);
    },
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) controller.close();
      else controller.enqueue(value);
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
  return { head, stream };
}
