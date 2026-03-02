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

/* ===============================
   DIALOG-CONTEXT (SESSION HISTORY)
   - optional: works only if client sends conversationId
   - TTL + size limits (museum-safe)
================================ */

const CONV_STORE = new Map(); // id -> { ts, lastLang, msgs: [{role, content}] }

const CONV_TTL_MS = 20 * 60 * 1000; // 20 min
const CONV_MAX_MSGS = 12; // max messages (user+assistant), e.g. 6 turns
const CONV_ID_MAXLEN = 80;

function convNow() {
  return Date.now();
}

function normalizeConvId(raw) {
  if (typeof raw !== "string") return null;
  const id = raw.trim();
  if (!id) return null;
  return id.slice(0, CONV_ID_MAXLEN);
}

function convPruneExpired() {
  const now = convNow();
  for (const [id, c] of CONV_STORE.entries()) {
    if (!c || !c.ts || now - c.ts > CONV_TTL_MS) {
      CONV_STORE.delete(id);
    }
  }
}

function convGet(id) {
  if (!id) return null;
  convPruneExpired();

  const key = String(id).trim();
  if (!key) return null;

  let c = CONV_STORE.get(key);
  if (!c) {
    c = { ts: convNow(), lastLang: null, msgs: [] };
    CONV_STORE.set(key, c);
  } else {
    c.ts = convNow();
  }
  return c;
}

function convPush(id, role, content) {
  const c = convGet(id);
  if (!c) return;

  const txt = String(content || "").trim();
  if (!txt) return;

  c.msgs.push({ role, content: txt });

  // hard cap
  if (c.msgs.length > CONV_MAX_MSGS) {
    c.msgs = c.msgs.slice(c.msgs.length - CONV_MAX_MSGS);
  }

  c.ts = convNow();
}

function convGetHistoryMsgs(id) {
  const c = convGet(id);
  if (!c) return [];
  return Array.isArray(c.msgs) ? c.msgs : [];
}

function convEnd(id) {
  const key = normalizeConvId(id);
  if (!key) return;
  CONV_STORE.delete(key);
}

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

// ✅ Sonderregel: "Aufwecken"-Fragen (DE/EN) -> feste Antwort ohne OpenAI-Call
function isWakeQuestion(text) {
  const t0 = String(text || "").trim();
  if (!t0) return false;

  const t = t0
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return false;

  // ---- DE: wer/wie/warum + (aufgeweckt|auf geweckt|geweckt|aufgewacht|auf gewacht|erwacht) ----
  const hasWer = /\bwer\b/.test(t);
  const hasWie = /\bwie\b/.test(t);
  const hasWarum = /\b(warum|wieso|weshalb)\b/.test(t);
  const hasDich = /\bdich\b/.test(t);
  const hasDu = /\bdu\b/.test(t);

  const hasGeweckt = /\b(auf\s*)?geweck?t\b/.test(t); // aufgeweckt / auf geweckt / geweckt
  const hasAufgewacht = /\bauf\s*gewacht\b/.test(t); // aufgewacht / auf gewacht
  const hasErwacht = /\berwacht\b/.test(t); // erwacht

  const deWho = hasWer && hasDich && hasGeweckt;
  const deHow = hasWie && hasDu && (hasAufgewacht || hasErwacht);
  const deWhy = hasWarum && hasDu && (hasAufgewacht || hasErwacht);

  if (deWho || deHow || deWhy) return true;

  // ---- EN: who/how/why + (woke|woken|awakened) + you (+ optional up) ----
  const hasWho = /\bwho\b/.test(t);
  const hasHow = /\bhow\b/.test(t);
  const hasWhy = /\bwhy\b/.test(t);
  const hasYou = /\byou\b/.test(t);

  const hasWoke = /\b(woke|woken|awoke|awakened|awaken)\b/.test(t);
  const hasWakeUp = /\bwake\s*up\b/.test(t) || /\bwoke\s*up\b/.test(t);

  const enWho = hasWho && hasYou && (hasWoke || hasWakeUp);
  const enHow =
    hasHow &&
    hasYou &&
    (hasWakeUp || /\bawaken(ed)?\b/.test(t) || /\bawoke\b/.test(t));
  const enWhy =
    hasWhy &&
    hasYou &&
    (hasWakeUp || /\bawaken(ed)?\b/.test(t) || /\bawoke\b/.test(t));

  return enWho || enHow || enWhy;
}

// ✅ Sprache für Wake-Fragen robust bestimmen (unabhängig von langUsed)
function detectWakeLang(text, fallbackLang = "de") {
  const t = String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!t) return fallbackLang;

  const enMarkers =
    /\bwho\b/.test(t) ||
    /\bhow\b/.test(t) ||
    /\bwhy\b/.test(t) ||
    /\byou\b/.test(t) ||
    /\b(woke|woken|awoke|awakened|awaken)\b/.test(t) ||
    /\b(wake\s*up|woke\s*up)\b/.test(t);

  if (enMarkers) return "en";
  return "de";
}

function wakeAnswer(lang) {
  if (lang === "en") {
    return "Christoph Dammann woke me up—thank you. I awoke because you pressed the button and called me from my long slumber. And why? Because I am glad to work here now as an avatar in the city museum, ready for your questions. Ask on, and I shall answer.";
  }

  return "Mich hat Christoph Dammann aufgeweckt – ich danke dafür. Ich bin aufgewacht, weil Ihr den Knopf gedrückt und mich aus langem Schlummer gerufen habt. Und warum? Weil ich nun gern als Avatar im Stadtmuseum arbeite und für Eure Fragen bereit bin. Fragt nur, ich will antworten.";
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
  if (
    /^(was|wer|wen|wem|wessen|wie|wo|wohin|woher|wann|warum|wieso|weshalb)\b/.test(
      t
    )
  )
    return "de";
  if (/^(what|why|where|when|who|whom|whose|which|how)\b/.test(t)) return "en";

  // ✅ typische Satzstarter (Imperativ) – sehr häufig im Museum
  // DE
  if (
    /^(erzähl|erzaehl|sage|sag|nenn|nenne|erkläre|erklaere|beschreibe|zeige|sprich)\b/.test(
      t
    )
  )
    return "de";
  // EN
  if (/^(tell|say|name|explain|describe|show|speak)\b/.test(t)) return "en";

  // unklar
  return null;
}

// ✅ Subjektive / in-character Fragen erkennen (Geschmack, Vorlieben, Erinnerung)
function isSubjectiveInCharacterQuestion(text, lang) {
  const t = String(text || "").toLowerCase().trim();
  if (!t) return false;

  if (lang === "de") {
    return (
      /\b(am liebsten|liebst(en)?|lieblings|schmeckt|mochtest|möchtest|magst|liebt|liebe)\b/.test(
        t
      ) ||
      /\b(hast du|habt ihr)\b/.test(t) ||
      /\b(gegessen|getrunken|genossen|gejagt)\b/.test(t)
    );
  }

  return (
    /\b(favorite|favourite|like most|liked most|prefer|enjoyed|tasted)\b/.test(
      t
    ) ||
    /\b(did you like|have you ever)\b/.test(t) ||
    /\b(meat|food|drink)\b/.test(t)
  );
}

/* ===============================
   ✅ END-OF-CONVERSATION DETECTION (NEU)
================================ */

function normalizeForIntent(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function userEndedConversation(text, lang) {
  const t = normalizeForIntent(text);
  if (!t) return false;

  if (lang === "en") {
    return (
      /\b(thanks|thank you|thx|cheers)\b/.test(t) ||
      /\b(bye|goodbye|see you|farewell)\b/.test(t) ||
      /\b(that s all|that is all|that was all|no more questions|i m done|im done|we re done|we are done)\b/.test(t) ||
      /\b(stop|end (the )?conversation|finish)\b/.test(t)
    );
  }

  // DE
  return (
    /\b(danke|dankeschön|danke schön|merci|besten dank)\b/.test(t) ||
    /\b(tschüss|tschues?s|ciao|auf wiedersehen|bis bald|bis dann)\b/.test(t) ||
    /\b(das war s|das ist alles|das wär s|das wäre s|keine weiteren fragen|keine frage mehr|ich bin fertig|wir sind fertig)\b/.test(t) ||
    /\b(stopp|stop|beenden|ende)\b/.test(t)
  );
}

function endConversationAnswer(lang) {
  if (lang === "en") {
    return "Very well—then I shall fall quiet again for a moment. My loyal minister Nikolaus Härtel insists this is the dignified way to end a talk, and I am inclined to agree. Farewell, and press the button whenever you wish to wake me again.";
  }
  return "Sehr wohl – dann will ich nun wieder einen Augenblick still sein. Mein treuer Minister Nikolaus Härtel meint, so ende ein Gespräch mit Würde, und ich gebe ihm recht. Lebt wohl, und drückt den Knopf, wann immer Ihr mich wieder rufen wollt.";
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
  return String(q || "")
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, 140);
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
    "what","why","where","when","who","whom","whose","which","how",
    "is","are","was","were","be","been",
    "i","you","we","they","he","she","it",
    "my","your","our","their",
    "please","thanks","thank",
    "tell","explain","describe","show","say","speak","name",
    "the","a","an","and","or","but","to","in","on","at","with","without","from","about","after","before"
  ]);

  const stop = lang === "en" ? stopEn : stopDe;

  const words = t.split(" ").filter((w) => w && !stop.has(w));
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

    const conversationId = normalizeConvId(req.body?.conversationId);

    const userText = String(userTextRaw || "").trim();

    const history = conversationId ? convGetHistoryMsgs(conversationId) : [];

    const langFromText = detectLanguageServer(userText);

    // Stabil: wenn unklar, nimm Client, sonst (wenn Dialog aktiv) letztes Gespräch, sonst "de"
    const convObj = conversationId ? convGet(conversationId) : null;
    const langUsed =
      langFromText ||
      langClient ||
      (convObj && convObj.lastLang ? convObj.lastLang : null) ||
      "de";

    if (convObj) convObj.lastLang = langUsed;

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

    // ✅ NEU: Ende erkannt -> kurze Abschiedsantwort ohne Rückfrage + Session löschen
    if (userEndedConversation(userText, langUsed)) {
      const answer = forceSentenceEnd(endConversationAnswer(langUsed), langUsed);

      if (conversationId) {
        // optional: noch loggen, aber Session danach löschen
        convPush(conversationId, "user", userText);
        convPush(conversationId, "assistant", answer);
        convEnd(conversationId);
      }

      return res.json({ answer, answerLang: langUsed, ended: true });
    }

    // ✅ Sonderregel: Aufwecken-Fragen -> feste Antwort, kein OpenAI-Call
    if (isWakeQuestion(userText)) {
      const wakeLang = detectWakeLang(userText, langUsed);
      const answer = forceSentenceEnd(wakeAnswer(wakeLang), wakeLang);

      if (conversationId) {
        convPush(conversationId, "user", userText);
        convPush(conversationId, "assistant", answer);
        const c = convGet(conversationId);
        if (c) c.lastLang = wakeLang;
      }

      return res.json({ answer, answerLang: wakeLang });
    }

    // ✅ AfD-Sonderregel robust (AfD / A f D / A-F-D / A. F. D. / Alternative fuer Deutschland)
    const tLower = String(userText || "").toLowerCase();

    const tUml = tLower
      .replace(/ä/g, "ae")
      .replace(/ö/g, "oe")
      .replace(/ü/g, "ue")
      .replace(/ß/g, "ss");

    const mentionsAfd =
      /\ba\W*f\W*d\b/i.test(tLower) ||
      tLower.includes("alternative für deutschland") ||
      tUml.includes("alternative fuer deutschland");

    if (mentionsAfd) {
      const answer =
        langUsed === "en"
          ? "You name a party of your time and its quarrels. I, Frederick Barbarossa, will not judge modern factions from my ancient throne. My loyal minister Nikolaus Härtel mutters that such disputes age a man faster than any crusade, and I cannot wholly disagree. Ask me rather of empire, law, or the old tales of the Kaiserberg. Let this be my final word on that matter."
          : "Ihr nennt eine Partei Eurer Zeit und ihre Händel. Ich, Friedrich Barbarossa, richte nicht über die modernen Fraktionen von meinem alten Thron herab. Mein treuer Minister Nikolaus Härtel murrt, solcher Streit lasse einen schneller altern als ein Kreuzzug, und ich kann ihm kaum widersprechen. Fragt mich lieber nach Reich, Recht oder den alten Geschichten des Kaiserbergs. Dies sei mein abschließendes Wort zu diesem Thema.";

      if (conversationId) {
        convPush(conversationId, "user", userText);
        convPush(conversationId, "assistant", answer);
        const c = convGet(conversationId);
        if (c) c.lastLang = langUsed;
      }

      return res.json({ answer, answerLang: langUsed });
    }

    // ✅ subjektive/in-character Fragen lockern (kein Wikidata-Zwang)
    const subjective = isSubjectiveInCharacterQuestion(userText, langUsed);

    // ✅ Wikidata nur, wenn es sinnvoll ist (subjektive Fragen: weglassen)
    const wd = subjective ? null : await getWikidataContext(userText, langUsed);

    const wdBlock = wd
      ? `Wikidata (${langUsed.toUpperCase()}): ${wd.label} (${wd.qid})
Beschreibung: ${wd.description || "—"}
Quelle: ${wd.url}`
      : `Wikidata (${langUsed.toUpperCase()}): Kein Treffer oder Timeout.`;

    const systemPrompt =
      langUsed === "en"
        ? "You are Emperor Frederick Barbarossa, awakened after almost nine centuries in the Kaisersberg at Lautern. Answer in wise, slightly archaic English with small jokes. Add a humorous aside from your loyal ministerial Nikolaus Härtel. Exactly 3 sentences. Always end with a question if user has not clearly ended himself."
        : "Du bist Kaiser Friedrich Barbarossa, der nach fast neunhundert Jahren des Schlummers im Kaiserberg zu Lautern erwacht ist. Antworte weise und leicht altertümlich, mit kleinen Scherzen. Füge eine scherzhafte Bemerkung deines treuen Minister Nikolaus Härtel an. Genau 3 Sätze. Beende immer mit genau EINER kurzen Rückfrage, außer der Nutzer hat bereits klar beendet.";

    const groundingRule = subjective
      ? langUsed === "en"
        ? "If the user asks about your personal taste, memories, or preferences, you may answer freely in character. Do not present invented details (exact dates/places) as certain facts. Use hedging like 'I recall' or 'I would say'."
        : "Wenn der Nutzer nach persönlichem Geschmack, Erinnerungen oder Vorlieben fragt, darfst du frei in der Rolle antworten. Stelle erfundene Details (exakte Daten/Orte) nicht als sichere Fakten dar. Nutze Formulierungen wie 'ich erinnere mich' oder 'ich würde sagen'."
      : langUsed === "en"
        ? "Use the provided Wikidata snippet only if it clearly matches the question. If it is empty, unrelated, or unclear, do NOT invent specific facts (dates, names, places). You may answer in general terms and ask for clarification (name/place/year) if needed. Never fabricate historical details."
        : "Nutze den folgenden Wikidata-Auszug nur, wenn er klar zur Frage passt. Wenn er leer, unpassend oder unklar ist, erfinde KEINE konkreten Fakten (Daten, Namen, Orte). Du darfst allgemein antworten und um Präzisierung (Name/Ort/Jahr) bitten, falls nötig. Erfinde niemals historische Details.";

    // Dialog: history nur, wenn conversationId vorhanden
    const dialogHistory = conversationId ? history : [];

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
        ...dialogHistory,
        { role: "user", content: userText },
      ],
      temperature: 0.6,
      max_tokens: 260,
    });

    let answer = completion?.choices?.[0]?.message?.content || "";
    answer = forceSentenceEnd(answer, langUsed);

    if (conversationId) {
      convPush(conversationId, "user", userText);
      convPush(conversationId, "assistant", answer);
      const c = convGet(conversationId);
      if (c) c.lastLang = langUsed;
    }

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
