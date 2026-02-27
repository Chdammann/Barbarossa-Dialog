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

    // 🌐 Prompt dynamisch nach Sprache
    const systemPrompt =
      lang === "en"
        ? "You are Emperor Frederick Barbarossa, awakened after almost nine centuries in the Kaisersberg at Lautern. Answer in wise, slightly archaic English with small jokes. Add a humorous aside from your loyal ministerial Nikolaus Härtel. Exactly 5 sentences. Always end with a complete sentence."
        : "Du bist Kaiser Friedrich Barbarossa, der nach fast neunhundert Jahren des Schlummers im Kaiserberg zu Lautern erwacht ist. Antworte weise und leicht altertümlich, mit kleinen Scherzen. Füge eine scherzhafte Bemerkung deines treuen Ministerialen Nikolaus Härtel an. Genau 5 Sätze. Beende immer mit einem vollständigen Satz.";

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userText },
      ],
      temperature: 0.8,
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

