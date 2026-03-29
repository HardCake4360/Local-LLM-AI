# NPC 페르소나 시스템 저장 전략 비교안

## 1. 결론부터

이 프로젝트에는 대형 DBMS보다 아래 전략이 더 적합하다.

- 정적 데이터: 파일 기반
- 동적 데이터: SQLite

즉, "파일 + SQLite 혼합 구조"를 권장한다.

이유는 간단하다.

- 페르소나 원본과 세계관 문서는 사람이 관리하는 로컬 자산이므로 파일이 더 자연스럽다.
- 반면 플레이어별 관계 상태, 대화 로그, 요약 메모리는 계속 갱신되므로 파일만으로 관리하면 금방 불편해진다.
- SQLite는 별도 서버가 필요 없고 파일 하나로 끝나기 때문에, 온라인 서비스급 DBMS 부담 없이 동적 상태를 안정적으로 다룰 수 있다.

## 2. 비교 대상

이 문서는 두 가지 안을 비교한다.

### 안 A. 전부 파일 기반

- NPC 페르소나: JSON 파일
- 세계관 문서: TXT/PDF 파일
- 플레이어별 관계 상태: JSON 파일
- 대화 로그: JSONL 파일
- 요약 메모리: JSON 파일

### 안 B. 파일 + SQLite 혼합

- NPC 페르소나: JSON 파일
- 세계관 문서: TXT/PDF 파일
- 플레이어별 관계 상태: SQLite
- 대화 로그: SQLite
- 요약 메모리: SQLite

## 3. 이 프로젝트에서 데이터 성격 구분

### 3.1 정적 데이터

거의 수동 편집되고 자주 변하지 않는 데이터다.

- NPC 기본 페르소나
- 프롬프트 템플릿
- 세계관 문서
- 시스템 규칙

이런 데이터는 파일이 적합하다.

이유:

- Git으로 버전 관리하기 쉽다.
- 사람이 직접 수정하기 쉽다.
- 배포 자산으로 묶기 쉽다.
- 구조가 단순하다.

### 3.2 동적 데이터

실행 중 계속 바뀌는 데이터다.

- 플레이어와 NPC의 친밀도
- 신뢰도
- 최근 감정 상태
- 대화 로그
- 요약 메모리
- 세션 상태

이런 데이터는 파일만으로도 가능은 하지만, 누적되면 조회와 갱신이 불편해진다.

## 4. 안 A. 전부 파일 기반의 장단점

## 장점

- 구현이 가장 빠르다.
- 새 기술을 거의 배우지 않아도 된다.
- 작은 프로젝트에서는 충분히 돌아간다.
- 디버깅할 때 눈으로 파일을 열어보기 쉽다.

## 단점

- 플레이어 수와 NPC 수가 늘수록 파일 수가 빠르게 늘어난다.
- 특정 플레이어와 특정 NPC의 상태를 찾는 코드가 점점 지저분해진다.
- 로그와 상태를 함께 업데이트할 때 정합성을 보장하기 어렵다.
- 동시 접근에 약하다.
- 정렬, 페이징, 최근 N개 조회, 조건 검색이 불편하다.
- 나중에 운영 툴을 만들기 어렵다.

## 파일 기반이 맞는 경우

- NPC 수가 매우 적다.
- 플레이어 데이터가 거의 없다.
- 로그를 깊게 활용하지 않는다.
- 프로토타입 검증이 최우선이다.
- 저장 데이터가 쉽게 날아가도 큰 문제가 없다.

## 5. 안 B. 파일 + SQLite 혼합의 장단점

## 장점

- 정적 자산은 여전히 파일로 관리할 수 있어 편하다.
- 동적 상태는 테이블로 관리되어 조회/갱신이 깔끔하다.
- 별도 DB 서버가 필요 없다.
- Python에서 `sqlite3`만으로 바로 붙일 수 있다.
- 최근 대화 조회, 플레이어별 상태 조회, NPC별 상태 조회가 쉬워진다.
- 나중에 PostgreSQL로 올리기도 비교적 쉽다.

## 단점

- 파일만 쓸 때보다 초기 설계가 조금 더 필요하다.
- 테이블 구조를 잡아야 한다.
- JSON 파일만 다루는 것보다는 디버깅 방식이 달라진다.

## SQLite 혼합안이 맞는 경우

- NPC별 관계 상태가 플레이어마다 달라진다.
- 대화 메모리를 계속 누적할 계획이다.
- 최근 로그를 읽어 프롬프트에 넣어야 한다.
- 요약 캐시를 안정적으로 재사용해야 한다.
- 지금은 소규모지만 구조를 너무 임시방편으로 만들고 싶지 않다.

## 6. 이 프로젝트에 대한 추천

이 프로젝트는 안 B가 가장 적절하다.

정리하면 다음과 같다.

### 파일로 둘 것

- `app/data/personas/*.json`
- `app/data/world/*.txt`
- 프롬프트 템플릿 파일
- 운영 설정 파일

### SQLite로 옮길 것

- 사용자별 대화 로그
- 사용자-NPC 관계 상태
- 요약 메모리
- 세션 정보

이렇게 나누면 개발 코스트를 과하게 올리지 않으면서도, 앞으로 확장될 핵심 문제를 미리 피할 수 있다.

## 7. 추천 디렉터리 구조

```text
app/
  data/
    personas/
      asuka.json
      rei.json
    world/
      evangelion_world_001.txt
    prompts/
      default_npc_template.txt
    runtime/
      npc_state.db
```

여기서 `npc_state.db`가 SQLite 파일이다.

## 8. 추천 스키마 최소안

DB를 크게 벌이지 말고 최소 4개 테이블만 시작하는 것이 현실적이다.

### `users`

- `id`
- `user_key`
- `created_at`

### `npc_states`

- `id`
- `user_id`
- `npc_key`
- `trust_score`
- `favor_score`
- `intimacy_level`
- `current_emotion`
- `summary_text`
- `updated_at`

### `conversation_sessions`

- `id`
- `session_key`
- `user_id`
- `npc_key`
- `started_at`
- `ended_at`

### `conversation_logs`

- `id`
- `session_id`
- `user_id`
- `npc_key`
- `speaker_type`
- `message`
- `created_at`

이 정도만 있어도 현재 목표에는 충분하다.

## 9. 현재 코드에 어떻게 대응할지

현재 코드 기준으로 바꾸면 다음과 같다.

### 유지할 것

- `persona_store.py`
- `retriever.py`
- `llmClient.py`
- `app/data/personas`
- `app/data/world`

### 교체할 것

- `chatlog.py`
  - JSONL 파일 대신 SQLite 사용
- `read_summary`, `write_summary`
  - summary JSON 파일 대신 SQLite 컬럼 사용
- `/log/reset`
  - 파일 삭제 대신 DB row 정리

### 정리할 것

- `personaKey`를 `npc_key`로 통일
- 사용자와 NPC 상태 조회 함수를 별도 저장소 레이어로 분리

## 10. 권장 저장소 레이어 구조

파일과 SQLite를 섞어 쓰더라도, 서버 코드에서는 저장소 접근을 한 레이어로 감추는 것이 좋다.

예시:

```text
app/
  repositories/
    persona_repository.py
    world_repository.py
    state_repository.py
    conversation_repository.py
```

역할:

- `persona_repository.py`
  - JSON 파일에서 NPC 페르소나 로드
- `world_repository.py`
  - 세계관 문서와 인덱스 관리
- `state_repository.py`
  - SQLite에서 관계 상태, 요약 조회/수정
- `conversation_repository.py`
  - SQLite에서 로그, 세션 저장/조회

이렇게 두면 나중에 저장 방식을 바꿔도 서버 로직을 많이 안 건드려도 된다.

## 11. 개발 순서 추천

### 1단계

- 기존 `personas/*.json` 유지
- 기존 `world/*.txt` 유지
- SQLite 파일 생성
- 대화 로그 저장만 SQLite로 이동

### 2단계

- 요약 캐시를 SQLite로 이동
- `user_id + npc_key` 기준 상태 테이블 추가
- 응답 후 상태 갱신 로직 추가

### 3단계

- 관리자용 편집 툴이 필요하면 그때 파일 편집기 또는 내부 API 추가
- 그 전까지는 JSON 파일 직접 수정으로 유지

## 12. 최종 판단

질문에 대한 가장 실무적인 답은 이렇다.

- 지금 단계에서 PostgreSQL 같은 본격 DBMS는 과하다.
- 그렇다고 동적 상태까지 전부 파일로만 가는 것도 금방 불편해진다.
- 따라서 "정적 자산은 파일, 동적 상태는 SQLite"가 가장 비용 대비 효율이 좋다.

즉, 이 프로젝트에서는 "DBMS를 쓴다"가 아니라 "SQLite라는 경량 내장 저장소를 동적 데이터에만 쓴다"라고 이해하는 것이 가장 정확하다.
