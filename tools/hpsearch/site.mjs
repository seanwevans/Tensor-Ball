// Serves the app to headless Chromium with every dependency vendored from
// node_modules. The shipped page pulls TensorFlow.js from jsDelivr and
// three/cannon-es from esm.sh; a search that re-fetched those per trial would
// be slower, less reproducible, and dependent on the sandbox's egress. The DOM
// itself is the repo's index.html, textually rewritten, so the element ids the
// Dashboard queries stay in sync with the real page.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const MODULES = join(HERE, "node_modules");

const IMPORT_MAP = `<script type="importmap">
{
  "imports": {
    "three": "/vendor/three/build/three.module.js",
    "https://esm.sh/three@0.132.2": "/vendor/three/build/three.module.js",
    "https://esm.sh/three@0.132.2/examples/jsm/controls/OrbitControls.js": "/vendor/three/examples/jsm/controls/OrbitControls.js",
    "https://esm.sh/cannon-es@0.19.0": "/vendor/cannon-es/dist/cannon-es.js"
  }
}
</script>
`;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
  ".map": "application/json"
};

// Swap the two CDN <script> tags for the vendored copies (SRI hashes go with
// them — they pin the CDN response, and there is no CDN here) and prepend the
// import map that redirects the ES module specifiers.
function harnessIndex(html) {
  const out = html
    .replace(
      /<script\s+src="https:\/\/cdn\.jsdelivr\.net\/npm\/@tensorflow\/tfjs@[^"]+"[\s\S]*?><\/script>/,
      '<script src="/vendor/tfjs/tf.min.js"></script>'
    )
    .replace(
      /<script\s+src="https:\/\/cdn\.jsdelivr\.net\/npm\/@tensorflow\/tfjs-backend-webgpu@[^"]+"[\s\S]*?><\/script>/,
      '<script src="/vendor/tfjs/tf-backend-webgpu.min.js"></script>'
    )
    .replace("<head>", "<head>\n" + IMPORT_MAP);
  if (out.includes("cdn.jsdelivr.net"))
    throw new Error("site.mjs: a jsDelivr script tag survived the rewrite");
  if (!out.includes("/vendor/tfjs/tf.min.js"))
    throw new Error("site.mjs: tfjs script tag was not rewritten");
  return out;
}

const VENDOR = {
  "/vendor/tfjs/tf.min.js": join(MODULES, "@tensorflow/tfjs/dist/tf.min.js"),
  "/vendor/tfjs/tf-backend-webgpu.min.js": join(
    MODULES,
    "@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.min.js"
  )
};

export async function startSite(scriptSource) {
  const index = harnessIndex(await readFile(join(ROOT, "index.html"), "utf8"));
  const css = await readFile(join(ROOT, "style.css"), "utf8");

  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost").pathname;
    const send = (body, type) => {
      res.writeHead(200, { "content-type": type, "cache-control": "no-store" });
      res.end(body);
    };
    try {
      if (url === "/" || url === "/index.html") return send(index, MIME[".html"]);
      if (url === "/script.js") return send(scriptSource, MIME[".js"]);
      if (url === "/style.css") return send(css, MIME[".css"]);
      if (VENDOR[url]) return send(await readFile(VENDOR[url]), MIME[".js"]);
      if (url.startsWith("/vendor/")) {
        const rel = normalize(url.slice("/vendor/".length));
        if (rel.startsWith("..")) throw new Error("traversal");
        const ext = rel.slice(rel.lastIndexOf("."));
        return send(await readFile(join(MODULES, rel)), MIME[ext] || "application/octet-stream");
      }
      res.writeHead(404).end("not found");
    } catch (e) {
      res.writeHead(500).end(String(e));
    }
  });

  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;
  return { port, origin: `http://127.0.0.1:${port}`, close: () => server.close() };
}
