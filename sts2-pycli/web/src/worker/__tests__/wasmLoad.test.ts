import { afterEach, describe, expect, it, vi } from "vitest";
import { engineResponse, NO_DECOMPRESSION } from "../wasmLoad";

/** A wasm-shaped payload: the `\0asm` magic, then enough bytes to span chunks. */
const WASM = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...Array.from({ length: 70_000 }, (_, i) => i % 251)]);

/** A fetch that answers `bytes` in `chunk`-sized pieces (1 = the magic split across reads). */
function fakeFetch(bytes: Uint8Array, chunk: number, init: ResponseInit = {}) {
  return vi.fn(async () => {
    let at = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(c) {
        if (at >= bytes.length) return c.close();
        c.enqueue(bytes.slice(at, at + chunk));
        at += chunk;
      },
    });
    return new Response(body, { status: 200, ...init });
  }) as unknown as typeof fetch & ReturnType<typeof vi.fn>;
}

const bytesOf = async (r: Response) => new Uint8Array(await r.arrayBuffer());

/** gzip `bytes` the way the build does (CompressionStream: no node types in the worker project). */
const gzip = (bytes: Uint8Array) =>
  bytesOf(new Response(new Response(bytes as BodyInit).body!.pipeThrough(new CompressionStream("gzip"))));

describe("engineResponse", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("inflates the gzipped engine into an application/wasm response", async () => {
    for (const chunk of [1, 3, 4096]) {
      const r = await engineResponse("/wb-assets/engine.wasm-abc.gz", fakeFetch(await gzip(WASM), chunk));
      expect(r.headers.get("content-type")).toBe("application/wasm");
      expect(await bytesOf(r)).toEqual(WASM);
    }
  });

  it("passes bytes a host already inflated (Content-Encoding: gzip) through", async () => {
    const r = await engineResponse("/wb-assets/engine.wasm-abc.gz", fakeFetch(WASM, 2));
    expect(r.headers.get("content-type")).toBe("application/wasm");
    expect(await bytesOf(r)).toEqual(WASM);
  });

  it("hands the plain dev-server wasm over untouched", async () => {
    const f = fakeFetch(WASM, 4096, { headers: { "content-type": "application/wasm" } });
    const r = await engineResponse("/src/wasm/pkg/sts2_workbench_wasm_bg.wasm?import", f);
    expect(await bytesOf(r)).toEqual(WASM);
  });

  it("says which browsers it needs when DecompressionStream is missing", async () => {
    vi.stubGlobal("DecompressionStream", undefined);
    await expect(engineResponse("/e.wasm.gz", fakeFetch(await gzip(WASM), 4096))).rejects.toThrow(NO_DECOMPRESSION);
  });

  it("fails loudly on a failed download", async () => {
    const f = vi.fn(async () => new Response("nope", { status: 404, statusText: "Not Found" })) as unknown as typeof fetch;
    await expect(engineResponse("/e.wasm.gz", f)).rejects.toThrow(/404/);
  });
});
