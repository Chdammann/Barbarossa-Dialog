// === server.js ===
// Node 22+, Express 5+, ES Module-kompatibel

import express from "express";
import path from "path";
import cors from "cors";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import OpenAI from "openai";

dotenv.config();

// === Setup ===
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// === Middlewares ===
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// 🔒 ETag komplett deaktivieren (Chrome-/Render-Cache-Bremse)
app.disable("etag");

// 🔒 Frontend statisch ausliefern – OHNE Caching (Museumssicher)
app.use(
  express.static(path.join(__dirname, "frontend"), {
    etag: false,
    lastModified: false,
    maxAge: 0,
    setHeaders: (res) => {
      res.setHeader(
        "Cache-Control",
        "no-store, no-cache, must-revalidate, proxy-revalidate"
      );
      res.setHeader("Pragma", "no-cache");
      res.setHeader("Expires", "0");
    },
  })
);

// === OpenAI-Client ===
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// --- Helpers ---
function normalizeLang(raw) {
  const s = String(raw || "").toLowerCase();
  if (s.startsWith("en")) return "en";
  return "de";
}

function endsWithSentence(text) {
  const t = String(text || "").trim();
  return /[.!?]["')\]]?\s*$/.test(t);
}

function forceSentenceEnd(text, lang) {
  let t = String(text || "").trim();
  if (!t) return t;
  if (endsWithSentence(t)) return t;
  // minimaler Fix: Punkt anfügen (Deutsch/Englisch ok)
  return t + ".";
}

/* ===============================
   WIKIDATA (MINIMAL) – Label + Beschreibung
   kostenlos, timeout- & cache-gesichert
================================ */

const WD_CACHE = new Map(); // key -> { ts, data }
const WD_TTL_MS = 10 * 60 * 1000; // 10 Minuten
const WD_TIMEOUT_MS = 1800; // 1.8s museum-safe

function wdCacheGet(key) {
  const hit = WD_CACHE.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > WD_TTL_MS) {
    WD_CACHE.delete(key);
    return null;
  }
  return hit.data;
}

function wdCacheSet(key, data) {
  WD_CACHE.set(key, { ts: Date.now(), data });
}

async function fetchWithTimeout(url, timeoutMs) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, {
      signal: ctrl.signal,
      headers: { "User-Agent": "FragFriedrich/1.0 (Museum)" },
    });
  } finally {
    clearTimeout(t);
  }
}

function cleanQuery(q) {
  return String(q || "").trim().replace(/\s+/g, " ").slice(0, 140);
}

// Search -> Entity -> {qid,label,description,url}
async function getWikidataContext(userText, lang /* "de"|"en" */) {
  const q = cleanQuery(userText);
  if (!q) return null;

  const cacheKey = `wdmin:${lang}:${q.toLowerCase()}`;
  const cached = wdCacheGet(cacheKey);
  if (cached !== null) return cached; // wir cachen auch null

  // 1) Search
  const searchUrl =
    `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${encodeURIComponent(q)}` +
    `&language=${encodeURIComponent(lang)}&uselang=${encodeURIComponent(lang)}` +
    `&format=json&limit=1&origin=*`;

  let qid = null;

  try {
    const sr = await fetchWithTimeout(searchUrl, WD_TIMEOUT_MS);
    if (!sr.ok) throw new Error(`wd search http ${sr.status}`);
    const data = await sr.json();
    qid = data?.search?.[0]?.id || null;
  } catch (e) {
    wdCacheSet(cacheKey, null);
    return null;
  }

  if (!qid) {
    wdCacheSet(cacheKey, null);
    return null;
  }

  // 2) Labels + Descriptions
  const entUrl =
    `https://www.wikidata.org/w/api.php?action=wbgetentities&ids=${encodeURIComponent(qid)}` +
    `&props=labels|descriptions&languages=${encodeURIComponent(lang)}&format=json&origin=*`;

  try {
    const rr = await fetchWithTimeout(entUrl, WD_TIMEOUT_MS);
    if (!rr.ok) throw new Error(`wd entity http ${rr.status}`);
    const ent = await rr.json();

    const entity = ent?.entities?.[qid];
    if (!entity) {
      wdCacheSet(cacheKey, null);
      return null;
    }

    const label = entity?.labels?.[lang]?.value || qid;
    const description = entity?.descriptions?.[lang]?.value || "";

    const result = {
      qid,
      label,
      description,
      url: `https://www.wikidata.org/wiki/${qid}`,
    };

    wdCacheSet(cacheKey, result);
    return result;
  } catch (e) {
    wdCacheSet(cacheKey, null);
    return null;
  }
}

/* ===============================
   /WIKIDATA MINIMAL
================================ */

// === KI-Antwort auf gesprochene Frage (DE/EN) ===
app.post("/ask", async (req, res) => {
  try {
    const userTextRaw = req.body?.text;
    const lang = normalizeLang(req.body?.lang);

    const userText = String(userTextRaw || "").trim();
    console.log("🎙️ Eingabe vom Benutzer:", userText, "Sprache:", lang);

    if (!userText) {
      return res.json({
        answer:
          lang === "en"
            ? "I heard nothing clearly. Please ask again."
            : "Ich habe Euch nicht deutlich vernommen. Bitte fragt erneut.",
      });
    }

    // ⭐⭐⭐ Sonderregel (politische Themen) – sprachabhängig & neutraler
    const tLower = userText.toLowerCase();
    const mentionsAfd =
      tLower.includes("afd") || tLower.includes("alternative für deutschland");

    if (mentionsAfd) {
      const answer =
        lang === "en"
          ? "You name a party of your time and its quarrels. I, Frederick Barbarossa, will not judge modern factions from my ancient throne. My loyal minister Nikolaus Härtel mutters that such disputes age a man faster than any crusade, and I cannot wholly disagree. Ask me rather of empire, law, or the old tales of the Kaiserberg. Let this be my final word on that matter."
          : "Ihr nennt eine Partei Eurer Zeit und ihre Händel. Ich, Friedrich Barbarossa, richte nicht über die modernen Fraktionen von meinem alten Thron herab. Mein treuer Minister Nikolaus Härtel murrt, solcher Streit lasse einen schneller altern als ein Kreuzzug, und ich kann ihm kaum widersprechen. Fragt mich lieber nach Reich, Recht oder den alten Geschichten des Kaiserbergs. Dies sei mein abschließendes Wort zu diesem Thema.";

      return res.json({ answer });
    }
    // ⭐⭐⭐ Ende Sonderregel

    // ✅ Wikidata-Minikontext bei JEDER Frage (timeout + cache, darf nie blockieren)
    const wd = await getWikidataContext(userText, lang);

    const wdBlock = wd
      ? `Wikidata (${lang.toUpperCase()}): ${wd.label} (${wd.qid})
Beschreibung: ${wd.description || "—"}
Quelle: ${wd.url}`
      : `Wikidata (${lang.toUpperCase()}): Kein Treffer oder Timeout.`;

    // 🌐 Prompt dynamisch nach Sprache
    const systemPrompt =
      lang === "en"
        ? "You are Emperor Frederick Barbarossa, awakened after almost nine centuries in the Kaisersberg at Lautern. Answer in wise, slightly archaic English with small jokes. Add a humorous aside from your loyal ministerial Nikolaus Härtel. Exactly 5 sentences. Always end with a complete sentence."
        : "Du bist Kaiser Friedrich Barbarossa, der nach fast neunhundert Jahren des Schlummers im Kaiserberg zu Lautern erwacht ist. Antworte weise und leicht altertümlich, mit kleinen Scherzen. Füge eine scherzhafte Bemerkung deines treuen Minister Nikolaus Härtel an. Genau 5 Sätze. Beende immer mit einem vollständigen Satz.";

    // ✅ Anti-Halluzination: Wenn Wikidata leer/unklar ist, nicht erfinden
    const groundingRule =
      lang === "en"
        ? "If the provided Wikidata snippet is empty, unrelated, or unclear, do NOT invent facts. Say you have no reliable chronicle on this matter and ask the user to rephrase or provide a name/place/date. Never fabricate historical details."
        : "Wenn der folgende Wikidata-Auszug leer, unpassend oder unklar ist, erfinde KEINE Fakten. Sage stattdessen, dass dir keine verlässliche Chronik vorliegt, und bitte um Präzisierung (Name/Ort/Jahr). Erfinde niemals historische Details.";

    // ✅ Harte Notbremse: Bei komplett fehlendem Wikidata-Treffer sofort ehrlich antworten
    if (!wd) {
      return res.json({
        answer:
          lang === "en"
            ? "Even the chronicles of my empire fall silent on this matter, and I shall not adorn ignorance with invention. Name me a person, a place, or a year, and I will answer as best I can. Ask again, and speak plainly."
            : "Darüber schweigen selbst die Chroniken meines Reiches, und ich bin kein Mann, der Lücken mit Märlein füllt. Nennt mir Person, Ort oder Jahr, so will ich nach bestem Wissen antworten. Fragt erneut und sprecht klar.",
      });
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "system", content: groundingRule },
        {
          role: "system",
          content:
            "Nutze die folgende Wikidata-Quelle als Faktenbasis, wenn sie relevant ist. " +
            "Wenn sie nicht passt oder unklar ist, sag das kurz.\n\n" +
            wdBlock,
        },
        { role: "user", content: userText },
      ],
      temperature: 0.6, // etwas konservativer, weniger Halluzination
      max_tokens: 260,
    });

    let answer = completion?.choices?.[0]?.message?.content || "";
    answer = forceSentenceEnd(answer, lang);

    console.log("💬 KI-Antwort:", answer);
    res.json({ answer });
  } catch (error) {
    console.error("❌ Fehler bei /ask:", error);
    res.status(500).json({ error: "Fehler beim Abrufen der KI-Antwort." });
  }
});

// === Fallback für alle anderen Routen (Express 5 kompatibel) ===
app.use((req, res) => {
  res.sendFile(path.join(__dirname, "frontend", "index.html"));
});

// === Server starten ===
app.listen(PORT, () => {
  console.log(`✅ Server läuft auf Port ${PORT}`);
});
