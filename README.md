# Chat With Docs (RAG)
## Tech Stack
| Layer             | Choice                          | Rationale                           |
| ----------------- | ------------------------------- | ----------------------------------- |
| Vector store      | **pgvector (managed Supabase)** | 1-click, UK region, SQL familiarity |
| Embeddings & chat | Azure OpenAI - gpt-4o           | Enterprise SLA, GDPR-aligned        |
| API               | FastAPI                         | Async, quick setup                  |
| Front-end         | Next.js + shadcn/ui             | Rapid UI, SSR                       |
| Auth              | Clerk.dev free tier             | Offload security; JWT passthrough   |
| Hosting           | Fly.io UK or Railway            | Minutes to deploy, EU data          |

## High level timeline
| Week                                 | Objectives & key deliverables                                                                                                                                                                                                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0 (Kick-off – 2 days)**<br>Day 1–2 | • Agree scope, pick stack (Postgres + pgvector or Chroma, LangChain, Azure OpenAI), set up Git repo, CI.<br>• Draft 1-page product spec **and** 3-slide sales pitch.<br>• Github Projects board created with sprint backlog.                                                                         |
| **1 (Prototype)**<br>Day 3–9         | **Backend:** PDF/Word ingest → text split → embeddings stored.<br>**UI:** Basic chat window pulling answers from RAG endpoint.<br>Validate privacy/compliance objections; line up 10 target firms; cold-outreach email template. *Deliverable*: Working local demo on sample contracts. |
| **2 (MVP + Security)**<br>Day 10–16  | **Backend:** JWT auth, per-firm data isolation, deletion endpoint, simple logging.<br>**UI:** Brand-aligned skin, upload drag-and-drop, answer citations.<br>Finish demo video, book 3 discovery calls, create pricing model (£199/user/month pilot).                                   |
| **3 (Pilot Ready)**<br>Day 17–23     | **Backend:** Rate-limit, basic admin dashboard, usage metrics.<br>**UI:** Polished error states, mobile-friendly.<br>Run 3 live demos, secure 1 signed pilot (2-week free trial); draft memorandum of understanding (NDA + pilot T\&Cs).                                                |
| **4 (Go-Live & GTM)**<br>Day 24–30   | Harden infra (vault secrets, backups), deploy on Fly.io/Linode UK (UK/EU data locality).<br>Launch landing page (Carrd + Stripe).<br>Collect pilot feedback, publish case study draft.<br>Set next-month roadmap only **after** revenue signal.                                                    |
