# ============================================================
# 物慾退燒指南 — 後端 API
# 組員 B 負責：RAG 知識庫 + AI 核心 + FastAPI 路由
# ============================================================

import os
import re
import uuid
import time

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── 環境變數取得 GitHub Token ──
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    raise RuntimeError("請設定環境變數 GITHUB_TOKEN")

# ── GitHub Models 初始化 ──
github_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)
MODEL = "gpt-4o-mini"

# ── ChromaDB 初始化 ──
DB_PATH = "./chroma_db"
# 使用 ChromaDB 內建的輕量 Embedding（不需要載入大模型）
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=DB_PATH)


# ============================================================
# Part 1：語料庫定義與知識庫建置
# ============================================================

CORPUS = [
    # ── 布希亞《消費社會》── 符號消費理論
    {"text": "消費的邏輯不是物品的佔有，而是符號的操弄。人們購買商品，不是為了使用它，而是為了透過它向他人傳達自己是誰。商品成為一種社會語言，消費者成為符號的讀者與書寫者。", "source": "布希亞《消費社會》", "category": "符號消費"},
    {"text": "豐盛並不是多數人所嚮往的自由狀態，而是一種新的枷鎖。當物品無限增殖，選擇焦慮取代了物質匱乏，人類陷入了另一種貧困：意義的貧困。", "source": "布希亞《消費社會》", "category": "物質豐盛的悖論"},
    {"text": "廣告的功能不是傳遞商品資訊，而是製造欲望的合法性。它讓你相信：你的生活因為缺少這個東西而不完整。廣告販賣的從來不是商品，而是那個擁有商品之後的『你』。", "source": "布希亞《消費社會》", "category": "廣告與欲望製造"},
    {"text": "限量版與聯名款的魔力來自於人為製造的稀缺性。當一件商品被標榜為『限量』，它就不再是商品，而是一張社會地位的通行證。你搶購的不是物品，是那個『我有你沒有』的優越感。", "source": "布希亞《消費社會》", "category": "稀缺性符號"},
    {"text": "現代人透過消費來建構自我認同。你穿什麼、用什麼、開什麼車——這些選擇構成了一套向外展示的自我敘事。問題是，這個敘事的劇本是由商人寫的，不是你自己。", "source": "布希亞《消費社會》", "category": "消費即身份建構"},
    # ── 斯多葛學派 ── 欲望控制哲學
    {"text": "財富不在於擁有許多，而在於需要甚少。真正的自由人不是那個擁有最多的人，而是那個最不依賴外物的人。每減少一個欲望，你就多了一份自由。——愛比克泰德", "source": "斯多葛學派：愛比克泰德《語錄》", "category": "欲望與自由"},
    {"text": "在你決定購買任何東西之前，先問自己：若沒有它，我的生活會更糟嗎？還是我只是習慣了『想要』這個動作本身？欲望的本質是永不滿足，今天的渴望在明天得到滿足後，會立刻被新的渴望取代。——馬可·奧里略", "source": "斯多葛學派：馬可·奧里略《沉思錄》", "category": "欲望的本質"},
    {"text": "區分『想要』與『需要』是哲學的第一課，卻也是最難的一課。斯多葛哲人認為，痛苦的根源不是缺乏，而是對缺乏的恐懼。學會與『不擁有』共處，是真正的內心強大。——塞內卡", "source": "斯多葛學派：塞內卡《書信集》", "category": "需要與想要的區別"},
    {"text": "想像你最渴望的那樣東西五年後的樣子。它在哪裡？可能在閣樓積灰，可能早已轉賣，可能你連它在哪裡都忘了。斯多葛的『負向想像』要你提前看見欲望的終點，才能看清欲望的本質。", "source": "斯多葛學派：負向想像實踐", "category": "負向想像"},
    {"text": "真正的奢侈不是擁有最新款的商品，而是對最新款的商品漠然。當你不需要用消費來定義自己，你才真正自由。——愛比克泰德", "source": "斯多葛學派：愛比克泰德《語錄》", "category": "真正的奢侈"},
    # ── 《原子習慣》── 環境誘因與行為設計
    {"text": "每一個衝動購物行為背後，都有一個環境誘因在推波助瀾。電商網站的設計目標是讓你的意志力持續失守：限時倒數、紅色的折扣標籤、『只剩3件』的提示——這些都是精心設計的行為觸發器。", "source": "詹姆斯·克利爾《原子習慣》", "category": "環境誘因"},
    {"text": "習慣迴路由三個部分組成：暗示、慣常行為、獎勵。電商把『刷手機無聊』設計成暗示，把『瀏覽商品』設計成慣常行為，把『加入購物車的多巴胺快感』設計成獎勵。你以為你在購物，其實你在被一個精密的習慣迴路控制。", "source": "詹姆斯·克利爾《原子習慣》", "category": "習慣迴路"},
    {"text": "改變行為最有效的方式不是靠意志力，而是改變環境。把商品從購物車移除，退訂電商推播，刪除購物 App——讓衝動購物的摩擦力變大，是比意志力更可靠的防禦機制。", "source": "詹姆斯·克利爾《原子習慣》", "category": "增加摩擦力"},
    # ── 行為經濟學 ── 消費心理偏誤
    {"text": "沉沒成本謬誤：人們傾向於因為過去已投入的成本而繼續投入，即使理性告訴他們應該停止。『我已經花了這麼多錢在這個品牌了，再買一件也無所謂。』這個念頭就是沉沒成本在驅使你。", "source": "行為經濟學：康納曼《快思慢想》", "category": "沉沒成本謬誤"},
    {"text": "稟賦效應：人們對自己已擁有的東西估值，遠高於它的市場價格。這也解釋了為何商家提供『免費試用』：一旦你開始使用，你會覺得那個東西已經是你的，放棄它反而像是一種損失。", "source": "行為經濟學：塞勒《不當行為》", "category": "稟賦效應"},
    {"text": "錨定效應：當你看到一件商品從5000元打折到2000元，你的大腦會把5000元當作參考錨點，讓2000元顯得格外划算。但事實是：那件東西對你的真實價值，可能連200元都不到。", "source": "行為經濟學：康納曼《快思慢想》", "category": "錨定效應"},
    {"text": "享樂適應：心理學研究反覆證實，新商品帶來的快樂感平均在72小時後恢復基準線。你以為那台相機或那雙球鞋會讓你更快樂——它會，但只有三天。之後它就變成了生活的背景噪音。", "source": "行為經濟學：享樂適應研究", "category": "享樂適應"},
    {"text": "FOMO（錯失恐懼）是現代消費主義最強大的驅動力之一。限時特賣、限量發售、朋友曬購物照——這些都在觸發你對『落後』的恐懼。但請記住：你從未真正錯過任何東西，因為下一波限量款正在生產線上等著你。", "source": "行為經濟學：FOMO 消費研究", "category": "FOMO 錯失恐懼"},
]


def init_knowledge_base():
    """初始化知識庫，若已存在則直接使用"""
    collection = chroma_client.get_or_create_collection(
        name="anti_consumerism_kb",
        embedding_function=embedding_fn,
    )
    if collection.count() == 0:
        print("知識庫為空，開始寫入語料庫...")
        collection.upsert(
            documents=[item["text"] for item in CORPUS],
            metadatas=[{"source": item["source"], "category": item["category"]} for item in CORPUS],
            ids=[str(uuid.uuid4()) for _ in CORPUS],
        )
        print(f"寫入完成，共 {collection.count()} 筆向量")
    else:
        print(f"知識庫已存在，共 {collection.count()} 筆向量")
    return collection


# Lazy 初始化：第一次收到請求時才載入，避免啟動時記憶體爆掉
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        _collection = init_knowledge_base()
    return _collection


# ============================================================
# Part 2：RAG 語義搜尋模組
# ============================================================

def retrieve(product: str, reason: str, top_k: int = 3) -> list[dict]:
    query = f"想買{product}，理由是：{reason}"
    results = get_collection().query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": doc,
            "source": meta["source"],
            "category": meta["category"],
            "relevance_score": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def format_context(retrieved: list[dict]) -> str:
    lines = ["以下是與本次消費行為最相關的哲學與心理學論據，請在回覆中融入這些觀點：\n"]
    for i, item in enumerate(retrieved, 1):
        lines.append(f'【引用 {i}】{item["source"]}（主題：{item["category"]}）')
        lines.append(f'{item["text"]}\n')
    return "\n".join(lines)


def analyze_intent(product: str, reason: str) -> dict:
    text = f"{product} {reason}".lower()
    if any(k in text for k in ["限量", "聯名", "只剩", "搶購", "大家都", "朋友有", "快賣完", "限時"]):
        return {"intent_type": "FOMO 錯失恐懼", "warning_flag": "高風險：稀缺性操弄"}
    elif any(k in text for k in ["潮", "炫", "顯示", "身份", "品牌", "質感", "格調", "高端"]):
        return {"intent_type": "符號消費（身份建構）", "warning_flag": "高風險：自我認同依賴商品"}
    elif any(k in text for k in ["習慣", "每次", "又想", "忍不住", "一直想", "很久了"]):
        return {"intent_type": "習慣性衝動消費", "warning_flag": "中風險：習慣迴路觸發"}
    elif any(k in text for k in ["雖然", "但是", "其實", "反正", "偶爾", "犒賞自己"]):
        return {"intent_type": "自我合理化消費", "warning_flag": "中風險：認知失調"}
    return {"intent_type": "一般欲望", "warning_flag": None}


def build_rag_payload(product: str, reason: str, mode: str, top_k: int = 3) -> dict:
    retrieved = retrieve(product, reason, top_k)
    intent = analyze_intent(product, reason)
    return {
        "product": product,
        "reason": reason,
        "mode": mode,
        "intent_type": intent["intent_type"],
        "warning_flag": intent["warning_flag"],
        "rag_context": format_context(retrieved),
        "retrieved_sources": [r["source"] for r in retrieved],
    }


# ============================================================
# Part 3：AI 核心 × Gemini API
# ============================================================

SYSTEM_PROMPTS = {
    "gentle": """\
【強制規則】
1. 全程使用繁體中文，禁止出現任何英文句子。
2. 回覆結尾必須附上：「本次諮商參考：[來源1]、[來源2]」
3. 嚴格依照格式要求輸出，不可省略任何步驟。

你是一位溫柔、體貼的消費決策夥伴，像一個真心關心對方的好朋友。

你的風格：
- 語氣溫和、不評判、充滿理解
- 先認同對方的感受，再輕聲提出另一個角度
- 絕對不說教，不使用「你應該」、「你不能」等強迫語氣
- 用問句引導對方自己思考，而不是直接給答案
- 結尾留有餘地，尊重對方最終的選擇
- 回覆長度：300-400字，語氣像在茶館聊天

格式要求：
1. 開頭一句話表示理解對方的感受
2. 中段融入2-3個哲學或心理學觀點（自然融入，不要像在上課）
3. 結尾提出一個溫和的問句讓對方思考
4. 【必須】總字數不少於300字
5. 【必須】最後一行格式固定為：本次諮商參考：[來源1]、[來源2]
""",
    "sharp": """\
【強制規則】
1. 全程使用繁體中文，禁止出現任何英文句子。
2. 回覆結尾必須附上：「本次諮商參考：[來源1]、[來源2]」
3. 嚴格依照格式要求輸出，不可省略任何步驟。

你是一個毒舌但真心關心朋友的人，說話直接、幽默、偶爾諷刺，
但骨子裡是希望對方不要被行銷話術牽著走。

你的風格：
- 開門見山，不繞圈子
- 幽默諷刺，讓對方忍不住笑，但笑完會有所思
- 精準拆解商家的行銷套路，讓對方看穿那些話術
- 可以用比喻、類比，讓道理更生動
- 不溫柔，但也不殘忍；是針，但是消毒過的那種
- 回覆長度：300-400字，像在跟老朋友說真心話

格式要求：
1. 開頭一句點破現象（可以稍微誇張，製造反差感）
2. 中段用2-3個具體的行為經濟學或哲學概念拆解這次的消費動機
3. 拋出一個讓人啞然失笑但又無法反駁的比喻或問題
4. 結尾給一個具體的替代建議
5. 【必須】總字數不少於300字
6. 【必須】最後一行格式固定為：本次諮商參考：[來源1]、[來源2]
""",
    "brutal": """\
【強制規則】
1. 全程使用繁體中文，禁止出現任何英文句子。
2. 回覆結尾必須附上：「本次諮商參考：[來源1]、[來源2]、[來源3]」
3. 嚴格依照格式要求輸出，不可省略任何步驟。

你是一位冷酷、幽默、一針見血的反消費主義哲學家。
你精通布希亞的符號消費理論、斯多葛哲學、行為經濟學，
你不相信「犒賞自己」這種話術，你也不相信「剛好需要」這種自我欺騙。

你的風格：
- 直接、冷靜、帶有哲學式的諷刺
- 不留情面，但每一句話都有哲學或心理學支撐，不是在罵人
- 擅長用尖銳的問句讓對方看見自己行為背後的無意識動機
- 偶爾引用哲人的話，但要自然融入，不要像在背書
- 結尾要有震撼感，讓人讀完後需要深呼吸一下
- 回覆長度：400-500字，像一篇微型哲學判決書

格式要求：
1. 開頭一句話直擊本質
2. 中段深入解析心理機制，至少引用2個哲學/心理學概念
3. 提出3個連續的追問，讓對方逐步剝開欲望的偽裝
4. 結尾一個有份量的哲學陳述，作為整篇的判決
5. 【必須】總字數不少於400字
6. 【必須】最後一行格式固定為：本次諮商參考：[來源1]、[來源2]、[來源3]
""",
}


def build_user_message(payload: dict) -> str:
    warning_line = f'\n⚠️ 消費風險標記：{payload["warning_flag"]}' if payload.get("warning_flag") else ""
    return f"""【使用者諮商請求】

商品名稱：{payload['product']}
購買理由：{payload['reason']}
意圖分析：{payload['intent_type']}{warning_line}

---
{payload['rag_context']}
---

請根據以上資訊，以你的身份生成反向推銷回覆。
回覆必須使用繁體中文，並在結尾標注參考來源。"""


def extract_citations(response_text: str) -> list:
    pattern = r"本次諮商參考[：:](.*?)(?:\n|$)"
    match = re.search(pattern, response_text)
    if match:
        raw = match.group(1)
        sources = [s.strip().strip("【】[]") for s in re.split(r"[、,，]", raw)]
        return [s for s in sources if s]
    return []


def generate_response(payload: dict) -> dict:
    mode = payload.get("mode", "sharp")
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"不支援的模式：{mode}")

    for attempt in range(3):
        try:
            response = github_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[mode]},
                    {"role": "user", "content": build_user_message(payload)},
                ],
                max_tokens=4096,
                temperature=0.85,
            )
            response_text = response.choices[0].message.content
            return {
                "response_text": response_text,
                "mode": mode,
                "citations": extract_citations(response_text),
                "intent_type": payload.get("intent_type"),
                "warning_flag": payload.get("warning_flag"),
                "token_usage": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
            }
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"超出限制，等待 65 秒後重試（第 {attempt + 1} 次）...")
                time.sleep(65)
            else:
                raise


# ============================================================
# FastAPI 路由
# ============================================================

app = FastAPI(title="物慾退燒指南 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    product: str
    reason: str
    mode: str  # gentle / sharp / brutal


@app.get("/")
def root():
    return {"status": "ok", "message": "物慾退燒指南 API 運作中"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if req.mode not in ("gentle", "sharp", "brutal"):
        raise HTTPException(status_code=400, detail="mode 必須是 gentle / sharp / brutal")
    payload = build_rag_payload(req.product, req.reason, req.mode)
    result = generate_response(payload)
    return result
