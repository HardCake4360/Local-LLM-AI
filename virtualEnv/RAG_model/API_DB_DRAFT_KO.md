# Unity 연동 NPC 페르소나 시스템 API 명세 및 DB 스키마 초안

## 1. 목표

이 문서는 유니티 클라이언트와 연동되는 NPC 대화 시스템을 위해, 서버 중심 구조의 API 명세와 DB 스키마 초안을 정의한다.

핵심 전제는 다음과 같다.

- 유니티 클라이언트는 `npc_key`만 전달한다.
- NPC의 페르소나 원본 데이터, 관계 상태, 장기 기억, 프롬프트 빌드 로직은 서버가 관리한다.
- 같은 NPC라도 플레이어마다 관계 상태와 기억은 달라질 수 있다.
- 서버는 응답 생성 이후 대화 로그와 요약 메모리를 업데이트한다.

즉, 서버가 NPC 정체성의 Source of Truth 역할을 맡고, 클라이언트는 최소한의 식별 정보와 플레이어 입력만 전송하는 구조를 목표로 한다.

## 2. 아키텍처 개요

```text
Unity Client
  -> 대화 요청(user_id, session_id, npc_key, message)
  -> API Server
      -> NPC 기본 페르소나 조회
      -> 플레이어-NPC 관계 상태 조회
      -> 최근 대화 로그/요약 조회
      -> 관련 세계관/RAG 컨텍스트 조회
      -> 프롬프트 조합
      -> LLM 응답 생성
      -> 로그 저장
      -> 관계 상태/메모리/요약 갱신
  -> Unity Client에 응답 반환
```

## 3. 핵심 도메인 모델

이 시스템은 크게 5개 개념으로 나뉜다.

- `NPC`: 캐릭터 자체의 정적 정의
- `Persona Version`: NPC의 말투, 성격, 제약 조건 등 프롬프트 빌드용 구조화 데이터
- `Player-NPC State`: 특정 플레이어와 특정 NPC 사이의 관계 상태
- `Conversation Log`: 실제 대화 원문 기록
- `Memory Summary`: 최근/장기 메모리를 압축한 상태 정보

## 4. API 설계 원칙

- 클라이언트는 가능하면 페르소나 원문을 모른다.
- API는 `npc_key`를 기준으로 동작한다.
- 대화 API는 단건 응답과 스트리밍 응답을 모두 고려한다.
- 상태 조회 API와 운영자용 관리 API를 분리한다.
- 페르소나 수정은 버전 관리 가능하도록 설계한다.

## 5. 클라이언트용 API 명세

### 5.1 NPC 대화 요청

`POST /v1/chat`

가장 핵심이 되는 API다. 유니티는 이 API를 통해 NPC와 대화한다.

#### Request Body

```json
{
  "user_id": "player_001",
  "session_id": "sess_abc123",
  "npc_key": "asuka",
  "message": "오늘 기분이 어때?",
  "context": {
    "location": "classroom",
    "quest_id": "quest_12",
    "time_of_day": "evening"
  },
  "options": {
    "stream": false,
    "max_tokens": 256
  }
}
```

#### 필드 설명

- `user_id`: 플레이어 식별자
- `session_id`: 대화 세션 식별자
- `npc_key`: NPC 식별 키
- `message`: 플레이어 발화
- `context`: 게임 상태에서 넘어오는 부가 정보
- `options.stream`: 스트리밍 여부
- `options.max_tokens`: 이번 응답 길이 제한

#### Response Body

```json
{
  "request_id": "req_20260323_0001",
  "npc_key": "asuka",
  "message": "흥, 갑자기 그런 걸 왜 물어보는 건데? 그래도 오늘은 나쁘지 않아.",
  "meta": {
    "persona_version": 3,
    "memory_used": true,
    "retrieved_context_count": 2,
    "relationship_state": {
      "trust": 0.62,
      "favor": 0.55,
      "intimacy": "medium"
    }
  }
}
```

### 5.2 NPC 대화 스트리밍 요청

`POST /v1/chat/stream`

스트리밍 기반 응답 API다. 현재 프로젝트의 `/ask-stream`을 확장하는 방향으로 볼 수 있다.

#### Request Body

`POST /v1/chat`과 동일한 구조를 사용한다.

#### Response

- `text/plain` chunked stream
- 또는 `text/event-stream` 기반 SSE

권장 형식은 SSE다.

예시:

```text
event: token
data: 흥,

event: token
data: 갑자기

event: done
data: {"request_id":"req_20260323_0001"}
```

### 5.3 NPC 기본 정보 조회

`GET /v1/npcs/{npc_key}`

유니티가 NPC 목록 표시나 대화 UI 초기화에 사용할 수 있다.

#### Response Body

```json
{
  "npc_key": "asuka",
  "display_name": "소류 아스카 랑그레이",
  "profile": {
    "role": "에반게리온 2호기 전속 파일럿",
    "description": "강한 자존심과 공격적인 말투를 지닌 파일럿"
  },
  "status": {
    "is_active": true
  }
}
```

주의:

- 클라이언트에는 프롬프트 원문 전체를 내리지 않는다.
- UI 표시용 최소 정보만 반환하는 것이 안전하다.

### 5.4 플레이어와 NPC 관계 상태 조회

`GET /v1/users/{user_id}/npcs/{npc_key}/state`

대화 UI에서 친밀도, 신뢰도, 최근 관계 변화 등을 노출하고 싶을 때 사용한다.

#### Response Body

```json
{
  "user_id": "player_001",
  "npc_key": "asuka",
  "relationship_state": {
    "trust": 0.62,
    "favor": 0.55,
    "intimacy": "medium",
    "last_interaction_at": "2026-03-23T20:15:11Z"
  },
  "memory_summary": {
    "summary": "플레이어는 최근 아스카를 여러 차례 격려했고, 아스카는 겉으로는 퉁명스럽지만 경계가 조금 낮아진 상태다.",
    "topics": ["훈련", "격려", "불안"]
  }
}
```

### 5.5 대화 로그 조회

`GET /v1/users/{user_id}/npcs/{npc_key}/conversations`

옵션:

- `session_id`
- `limit`
- `cursor`

#### Response Body

```json
{
  "items": [
    {
      "conversation_id": "conv_1001",
      "session_id": "sess_abc123",
      "speaker": "user",
      "message": "오늘 기분이 어때?",
      "created_at": "2026-03-23T20:15:00Z"
    },
    {
      "conversation_id": "conv_1002",
      "session_id": "sess_abc123",
      "speaker": "npc",
      "message": "흥, 갑자기 그런 걸 왜 물어보는 건데?",
      "created_at": "2026-03-23T20:15:02Z"
    }
  ],
  "next_cursor": null
}
```

### 5.6 대화 리셋

`POST /v1/users/{user_id}/npcs/{npc_key}/reset`

기능:

- 최근 대화 로그 초기화
- 세션 상태 초기화
- 필요 시 memory summary만 유지하거나 삭제하는 옵션 지원

#### Request Body

```json
{
  "reset_scope": "session_only"
}
```

가능 값 예시:

- `session_only`
- `conversation_and_summary`
- `all_state`

## 6. 운영/관리자용 API 명세

운영자나 내부 툴에서 사용할 API다. 유니티 클라이언트에 직접 노출할 필요는 없다.

### 6.1 NPC 등록

`POST /admin/v1/npcs`

```json
{
  "npc_key": "asuka",
  "display_name": "소류 아스카 랑그레이",
  "world_id": "eva_world",
  "is_active": true
}
```

### 6.2 NPC 페르소나 버전 등록

`POST /admin/v1/npcs/{npc_key}/persona-versions`

```json
{
  "version": 3,
  "is_active": true,
  "persona_data": {
    "identity": {
      "name": "소류 아스카 랑그레이",
      "role": "에반게리온 2호기 전속 파일럿"
    },
    "personality": {
      "big_five": {
        "O": 65,
        "C": 70,
        "E": 85,
        "A": 30,
        "N": 80
      }
    },
    "linguistic_style": {
      "tone": "직설적이고 공격적이며 자신만만한 말투"
    },
    "behavioral_constraints": {
      "avoid_meta": true,
      "avoid_out_of_world_info": true
    }
  },
  "prompt_template_id": "default_npc_template_v1"
}
```

### 6.3 NPC 비활성화

`PATCH /admin/v1/npcs/{npc_key}`

```json
{
  "is_active": false
}
```

### 6.4 프롬프트 템플릿 관리

`POST /admin/v1/prompt-templates`

```json
{
  "template_key": "default_npc_template_v1",
  "system_prompt_template": "너는 {npc_name}이며, 다음 성격 규칙을 따른다: ...",
  "user_prompt_template": "상황: {game_context}\n질문: {user_message}"
}
```

## 7. 권장 응답 생성 플로우

`POST /v1/chat` 처리 시 서버 내부 플로우는 아래를 권장한다.

1. `npc_key`로 활성 NPC 확인
2. 활성 페르소나 버전 조회
3. `user_id + npc_key` 기준 관계 상태 조회
4. 최근 대화 로그와 메모리 요약 조회
5. 게임 컨텍스트와 RAG 검색 결과 조회
6. 프롬프트 템플릿에 구조화 데이터를 주입해 최종 prompt 생성
7. LLM 응답 생성
8. 대화 로그 저장
9. 관계 상태 및 메모리 요약 업데이트
10. 응답 반환

## 8. DB 스키마 초안

처음부터 복잡한 분산 DB까지 갈 필요는 없고, 초안 단계에서는 PostgreSQL 또는 SQLite에서 시작해도 된다. 다만 실제 서비스 확장을 생각하면 PostgreSQL 기준으로 설계하는 편이 무난하다.

## 9. 테이블 설계

### 9.1 `npcs`

NPC의 기본 메타 정보를 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `npc_key` | VARCHAR(100) UNIQUE | 외부에서 쓰는 NPC 식별 키 |
| `display_name` | VARCHAR(200) | 표시 이름 |
| `world_id` | VARCHAR(100) | 세계관 구분 키 |
| `description` | TEXT | UI 표시용 설명 |
| `is_active` | BOOLEAN | 사용 여부 |
| `created_at` | TIMESTAMP | 생성 시각 |
| `updated_at` | TIMESTAMP | 수정 시각 |

### 9.2 `npc_persona_versions`

NPC의 실제 페르소나 구조화 데이터를 버전별로 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `npc_id` | BIGINT FK -> npcs.id | 대상 NPC |
| `version` | INTEGER | 버전 번호 |
| `is_active` | BOOLEAN | 현재 활성 버전 여부 |
| `persona_data` | JSONB | 성격/말투/제약 등 구조화 데이터 |
| `prompt_template_id` | BIGINT FK | 사용할 프롬프트 템플릿 |
| `created_by` | VARCHAR(100) | 수정자 |
| `created_at` | TIMESTAMP | 생성 시각 |

권장 제약:

- `UNIQUE (npc_id, version)`
- 한 NPC당 `is_active = true`는 하나만 유지

### 9.3 `prompt_templates`

프롬프트 템플릿 정의를 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `template_key` | VARCHAR(100) UNIQUE | 템플릿 식별 키 |
| `system_prompt_template` | TEXT | 시스템 프롬프트 템플릿 |
| `user_prompt_template` | TEXT | 유저 프롬프트 템플릿 |
| `is_active` | BOOLEAN | 사용 여부 |
| `created_at` | TIMESTAMP | 생성 시각 |

### 9.4 `users`

플레이어 식별을 위한 기본 테이블이다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `user_key` | VARCHAR(100) UNIQUE | 외부 플레이어 ID |
| `created_at` | TIMESTAMP | 생성 시각 |

### 9.5 `player_npc_states`

플레이어와 NPC의 관계 상태를 저장한다. 실제 게임 경험 차별화의 핵심 테이블이다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `user_id` | BIGINT FK -> users.id | 플레이어 |
| `npc_id` | BIGINT FK -> npcs.id | NPC |
| `trust_score` | NUMERIC(4,3) | 신뢰도 0.000~1.000 |
| `favor_score` | NUMERIC(4,3) | 호감도 0.000~1.000 |
| `intimacy_level` | VARCHAR(20) | low/medium/high |
| `current_emotion` | VARCHAR(50) | 현재 NPC 감정 상태 |
| `memory_summary_id` | BIGINT FK | 최근 메모리 요약 참조 |
| `last_interaction_at` | TIMESTAMP | 최근 상호작용 시각 |
| `created_at` | TIMESTAMP | 생성 시각 |
| `updated_at` | TIMESTAMP | 수정 시각 |

권장 제약:

- `UNIQUE (user_id, npc_id)`

### 9.6 `conversation_sessions`

대화 세션 단위를 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `session_key` | VARCHAR(120) UNIQUE | 외부 세션 ID |
| `user_id` | BIGINT FK -> users.id | 플레이어 |
| `npc_id` | BIGINT FK -> npcs.id | NPC |
| `started_at` | TIMESTAMP | 시작 시각 |
| `ended_at` | TIMESTAMP NULL | 종료 시각 |
| `status` | VARCHAR(20) | active/closed |

### 9.7 `conversation_logs`

실제 발화를 순서대로 기록한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `session_id` | BIGINT FK -> conversation_sessions.id | 세션 |
| `user_id` | BIGINT FK -> users.id | 플레이어 |
| `npc_id` | BIGINT FK -> npcs.id | NPC |
| `speaker_type` | VARCHAR(20) | user/npc/system |
| `speaker_key` | VARCHAR(100) | 필요 시 npc_key 또는 user_key |
| `message` | TEXT | 실제 발화 |
| `tokens_in` | INTEGER NULL | 입력 토큰 수 |
| `tokens_out` | INTEGER NULL | 출력 토큰 수 |
| `created_at` | TIMESTAMP | 생성 시각 |

인덱스 권장:

- `(user_id, npc_id, created_at DESC)`
- `(session_id, created_at ASC)`

### 9.8 `memory_summaries`

대화 로그를 압축한 요약 메모리를 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `user_id` | BIGINT FK -> users.id | 플레이어 |
| `npc_id` | BIGINT FK -> npcs.id | NPC |
| `session_id` | BIGINT FK -> conversation_sessions.id NULL | 세션 단위 요약이면 참조 |
| `summary_type` | VARCHAR(30) | recent/long_term/relationship |
| `summary_text` | TEXT | 요약 본문 |
| `summary_json` | JSONB | 구조화된 메모리 객체 |
| `source_message_count` | INTEGER | 요약 대상 메시지 수 |
| `created_at` | TIMESTAMP | 생성 시각 |
| `updated_at` | TIMESTAMP | 수정 시각 |

### 9.9 `world_documents`

RAG 검색용 문서 메타 정보를 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `world_id` | VARCHAR(100) | 세계관 구분 키 |
| `document_key` | VARCHAR(100) UNIQUE | 문서 식별 키 |
| `title` | VARCHAR(200) | 제목 |
| `source_type` | VARCHAR(30) | txt/pdf/manual |
| `content` | TEXT | 원문 또는 원문 참조 |
| `is_active` | BOOLEAN | 활성 여부 |
| `created_at` | TIMESTAMP | 생성 시각 |

### 9.10 `world_document_chunks`

임베딩 검색 대상 청크를 저장한다.

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL PK | 내부 ID |
| `document_id` | BIGINT FK -> world_documents.id | 원본 문서 |
| `chunk_index` | INTEGER | 청크 순서 |
| `chunk_text` | TEXT | 청크 텍스트 |
| `embedding_ref` | VARCHAR(200) NULL | 벡터 저장소 참조 |
| `created_at` | TIMESTAMP | 생성 시각 |

## 10. 권장 SQL 초안

아래는 PostgreSQL 기준의 축약 DDL 예시다.

```sql
CREATE TABLE npcs (
  id BIGSERIAL PRIMARY KEY,
  npc_key VARCHAR(100) NOT NULL UNIQUE,
  display_name VARCHAR(200) NOT NULL,
  world_id VARCHAR(100),
  description TEXT,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE prompt_templates (
  id BIGSERIAL PRIMARY KEY,
  template_key VARCHAR(100) NOT NULL UNIQUE,
  system_prompt_template TEXT NOT NULL,
  user_prompt_template TEXT,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE npc_persona_versions (
  id BIGSERIAL PRIMARY KEY,
  npc_id BIGINT NOT NULL REFERENCES npcs(id),
  version INTEGER NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT FALSE,
  persona_data JSONB NOT NULL,
  prompt_template_id BIGINT REFERENCES prompt_templates(id),
  created_by VARCHAR(100),
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  UNIQUE (npc_id, version)
);

CREATE TABLE users (
  id BIGSERIAL PRIMARY KEY,
  user_key VARCHAR(100) NOT NULL UNIQUE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE conversation_sessions (
  id BIGSERIAL PRIMARY KEY,
  session_key VARCHAR(120) NOT NULL UNIQUE,
  user_id BIGINT NOT NULL REFERENCES users(id),
  npc_id BIGINT NOT NULL REFERENCES npcs(id),
  started_at TIMESTAMP NOT NULL DEFAULT NOW(),
  ended_at TIMESTAMP NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'active'
);

CREATE TABLE memory_summaries (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id),
  npc_id BIGINT NOT NULL REFERENCES npcs(id),
  session_id BIGINT NULL REFERENCES conversation_sessions(id),
  summary_type VARCHAR(30) NOT NULL,
  summary_text TEXT NOT NULL,
  summary_json JSONB,
  source_message_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE player_npc_states (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id),
  npc_id BIGINT NOT NULL REFERENCES npcs(id),
  trust_score NUMERIC(4,3) NOT NULL DEFAULT 0.500,
  favor_score NUMERIC(4,3) NOT NULL DEFAULT 0.500,
  intimacy_level VARCHAR(20) NOT NULL DEFAULT 'medium',
  current_emotion VARCHAR(50),
  memory_summary_id BIGINT NULL REFERENCES memory_summaries(id),
  last_interaction_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, npc_id)
);

CREATE TABLE conversation_logs (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT NOT NULL REFERENCES conversation_sessions(id),
  user_id BIGINT NOT NULL REFERENCES users(id),
  npc_id BIGINT NOT NULL REFERENCES npcs(id),
  speaker_type VARCHAR(20) NOT NULL,
  speaker_key VARCHAR(100),
  message TEXT NOT NULL,
  tokens_in INTEGER,
  tokens_out INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## 11. 최소 구현 단계 제안

처음부터 모든 기능을 다 만들기보다 아래 순서가 현실적이다.

### 단계 1

- `npcs`
- `npc_persona_versions`
- `users`
- `conversation_sessions`
- `conversation_logs`
- `POST /v1/chat`

이 단계에서는 관계 상태 없이도 기본 NPC 대화는 가능하다.

### 단계 2

- `player_npc_states`
- `memory_summaries`
- `/state` 조회 API
- 응답 후 요약 갱신

이 단계에서 플레이어별 개별 반응과 기억이 생긴다.

### 단계 3

- `world_documents`
- `world_document_chunks`
- 벡터 검색 연동
- 관리자용 페르소나/문서 관리 API

이 단계에서 RAG와 운영툴이 붙는다.

## 12. 현재 프로젝트에 바로 대응시키는 방법

현재 코드 기준으로 매핑하면 다음과 같이 옮겨갈 수 있다.

- `persona_store.py`
  - JSON 파일 로드 방식에서 `npc_persona_versions` 조회 방식으로 전환
- `chatlog.py`
  - JSONL 파일 저장 방식에서 `conversation_logs`, `memory_summaries` 저장 방식으로 전환
- `server.py`
  - `personaKey`를 `npc_key`로 명확히 통일
  - `/ask-stream`을 `/v1/chat/stream`으로 정리
  - 요약 캐시를 DB 조회 기반으로 교체
- `retriever.py`
  - 정적 파일 인덱스에서 `world_documents` 관리 체계로 확장

## 13. 결론

이 구조에서 가장 중요한 설계 포인트는 두 가지다.

- 클라이언트는 `npc_key` 중심의 얇은 인터페이스만 가진다.
- 서버는 NPC 페르소나, 관계 상태, 메모리, 프롬프트 조립을 전부 책임진다.

이렇게 해야 NPC 수가 늘어나도 관리가 가능하고, 플레이어별 상호작용 차별화도 자연스럽게 지원할 수 있다.
