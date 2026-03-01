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
  return t + ".";
}

// ✅ ROBUST: Sprache primär über Satzanfang bestimmen (Fragewörter + Imperativ-Starter)
function detectLanguageServer(text) {
  const t0 = String(text || "").trim();
  if (!t0) return null;

  const t = t0
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return null;

  // harte DE-Indikatoren
  if (/[äöüß]/.test(t0)) return "de";

  // harte Satzanfang-Regeln (Fragewörter)
  if (/^(was|wer|wen|wem|wessen|wie|wo|wohin|woher|wann|warum|wieso|weshalb)\b/.test(t)) return "de";
  if (/^(what|why|where|when|who|whom|whose|which|how)\b/.test(t)) return "en";

  // ✅ NEU: typische Satzstarter (Imperativ) – sehr häufig im Museum
  // DE
  if (/^(erzähl|erzaehl|sage|sag|nenn|nenne|erkläre|erklaere|beschreibe|zeige|sprich)\b/.test(t)) return "de";
  // EN
  if (/^(tell|say|name|explain|describe|show|speak)\b/.test(t)) return "en";

  // unklar
  return null;
}

/* ===============================
   WIKIDATA (MINIMAL) – Label + Beschreibung
================================ */

const WD_CACHE = new Map();
const WD_TTL_MS = 10 * 60 * 1000;
const WD_TIMEOUT_MS = 1800;

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

function toEntityQuery(userText, lang) {
  const t = String(userText || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return "";

  const stopDe = new Set([
    "wer","wen","wem","wessen","was","wie","wo","wohin","woher","wann","warum","wieso","weshalb",
    "ist","sind","war","waren","sei","seid","bin","bist",
    "ich","du","ihr","wir","sie","er","es",
    "mein","dein","euer","unser",
    "bitte","danke",
    "erkläre","erzähle","beschreibe","zeige","sage","sprich","nenn","nenne",
    "der","die","das","ein","eine","einen","einem","einer","und","oder","aber","zu","zum","zur",
    "im","in","am","an","auf","mit","ohne","von","für","über","nach","vor"
  ]);

  const stopEn = new Set([
    "what","why","where","when","who","whom","whose","which|how",
    "is","are","was","were","be","been",
    "i","you","we","they","he","she","it",
    "my","your","our","their",
    "please","thanks","thank",
    "tell","explain","describe","show","say","speak","name",
    "the","a","an","and","or","but","to","in","on","at","with","without","from","about","after","before"
  ]);

  const stop = (lang === "en") ? stopEn : stopDe;

  const words = t.split(" ").filter(w => w && !stop.has(w));
  return words.slice(0, 6).join(" ").slice(0, 80);
}

async function getWikidataContext(userText, lang) {
  const qRaw = cleanQuery(userText);
  if (!qRaw) return null;

  const q = toEntityQuery(qRaw, lang) || qRaw;
  const cacheKey = `wdmin:${lang}:${q.toLowerCase()}`;

  const cached = wdCacheGet(cacheKey);
  if (cached !== null) return cached;

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

// === KI-Antwort auf gesprochene Frage (DE/EN) ===
app.post("/ask", async (req, res) => {
  try {
    const userTextRaw = req.body?.text;

    const hasClientLang =
      typeof req.body?.lang === "string" && req.body.lang.trim().length > 0;
    const langClient = hasClientLang ? normalizeLang(req.body.lang) : null;

    const userText = String(userTextRaw || "").trim();

    const langFromText = detectLanguageServer(userText);
    const langUsed = langFromText || langClient || "de";

    console.log("🎙️ Eingabe vom Benutzer:", userText, "Sprache:", langUsed);

    if (!userText) {
      return res.json({
        answer:
          langUsed === "en"
            ? "I heard nothing clearly. Please ask again."
            : "Ich habe Euch nicht deutlich vernommen. Bitte fragt erneut.",
        answerLang: langUsed,
      });
    }

    const tLower = userText.toLowerCase();
    const mentionsAfd =
      tLower.includes("afd") || tLower.includes("alternative für deutschland");

    if (mentionsAfd) {
      const answer =
        langUsed === "en"
          ? "You name a party of your time and its quarrels. I, Frederick Barbarossa, will not judge modern factions from my ancient throne. My loyal minister Nikolaus Härtel mutters that such disputes age a man faster than any crusade, and I cannot wholly disagree. Ask me rather of empire, law, or the old tales of the Kaiserberg. Let this be my final word on that matter."
          : "Ihr nennt eine Partei Eurer Zeit und ihre Händel. Ich, Friedrich Barbarossa, richte nicht über die modernen Fraktionen von meinem alten Thron herab. Mein treuer Minister Nikolaus Härtel murrt, solcher Streit lasse einen schneller altern als ein Kreuzzug, und ich kann ihm kaum widersprechen. Fragt mich lieber nach Reich, Recht oder den alten Geschichten des Kaiserbergs. Dies sei mein abschließendes Wort zu diesem Thema.";

      return res.json({ answer, answerLang: langUsed });
    }

    const wd = await getWikidataContext(userText, langUsed);

    const wdBlock = wd
      ? `Wikidata (${langUsed.toUpperCase()}): ${wd.label} (${wd.qid})
Beschreibung: ${wd.description || "—"}
Quelle: ${wd.url}`
      : `Wikidata (${langUsed.toUpperCase()}): Kein Treffer oder Timeout.`;

    const systemPrompt =
      langUsed === "en"
        ? "You are Emperor Frederick Barbarossa, awakened after almost nine centuries in the Kaisersberg at Lautern. Answer in wise, slightly archaic English with small jokes. Add a humorous aside from your loyal ministerial Nikolaus Härtel. Exactly 5 sentences. Always end with a complete sentence."
        : "Du bist Kaiser Friedrich Barbarossa, der nach fast neunhundert Jahren des Schlummers im Kaiserberg zu Lautern erwacht ist. Antworte weise und leicht altertümlich, mit kleinen Scherzen. Füge eine scherzhafte Bemerkung deines treuen Minister Nikolaus Härtel an. Genau 5 Sätze. Beende immer mit einem vollständigen Satz.";

    const groundingRule =
      langUsed === "en"
        ? "Use the provided Wikidata snippet only if it clearly matches the question. If it is empty, unrelated, or unclear, do NOT invent specific facts (dates, names, places). You may answer in general terms and ask for clarification (name/place/year) if needed. Never fabricate historical details."
        : "Nutze den folgenden Wikidata-Auszug nur, wenn er klar zur Frage passt. Wenn er leer, unpassend oder unklar ist, erfinde KEINE konkreten Fakten (Daten, Namen, Orte). Du darfst allgemein antworten und um Präzisierung (Name/Ort/Jahr) bitten, falls nötig. Erfinde niemals historische Details.";

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
      temperature: 0.6,
      max_tokens: 260,
    });

    let answer = completion?.choices?.[0]?.message?.content || "";
    answer = forceSentenceEnd(answer, langUsed);

    console.log("💬 KI-Antwort:", answer);
    res.json({ answer, answerLang: langUsed });
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
