// === server.js ===
// Node 22+, Express 5+, ES Module-kompatibel

import express from "express";
import path from "path";
import cors from "cors";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import OpenAI from "openai";

dotenv.config(); // .env einlesen

// === Setup ===
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Port von Render (wird automatisch bereitgestellt) oder 3000 für lokale Entwicklung
const PORT = process.env.PORT || 3000;

// === Middlewares ===
app.use(cors());
app.use(express.json());

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

// === KI-Antwort auf gesprochene Frage ===
app.post("/ask", async (req, res) => {
  try {
    const userText = req.body.text;
    console.log("🎙️ Eingabe vom Benutzer:", userText);

    // ⭐⭐⭐ NEU: Diplomatische, leicht humorvolle AfD-Sonderregel
    if (
      userText?.toLowerCase().includes("afd") ||
      userText?.toLowerCase().includes("alternative für deutschland")
    ) {
      return res.json({
        answer:
          "Wohlan, Ihr sprecht von der AfD! Die Gelehrten Eurer Zeit stufen Teile dieser Partei als gesichert rechtsextrem ein. Doch ich, Friedrich Barbarossa, mische mich nicht weiter in die politischen Händel Eurer Neuzeit ein. Mein treuer Ministerialer Friedrich Februar meint zwar, ich solle mich lieber wieder in den Kaiserberg zurückziehen, um dem Streit aus dem Wege zu gehen, doch ich lächle nur milde. Dies sei meine abschließende Rede zu diesem Thema.",
      });
    }
    // ⭐⭐⭐ Ende der Sonderregel

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "Du bist Kaiser Friedrich Barbarossa, der nach fast neunhundert Jahren des Schlummers im Kaiserberg zu Lautern erwacht ist. Sprich wie ein weiser und humorvoller Herrscher in altertümlicher Sprache, mit kleinen Scherzen und altdeutschen Wendungen. Füge noch eine scherzhafte Meinung deines treuen Minnisterialen Friedrich Februar an. Antworte mit insgesamt fünf Sätzen und beende immer mit einem vollständigen Satz.",
        },
        { role: "user", content: userText },
      ],
      temperature: 0.8,
      max_tokens: 250,
    });

    const answer = completion.choices[0].message.content;
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



