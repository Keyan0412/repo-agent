# repo-agent

This is an EvidenceGraph-driven repo analysis agent.

本项目不是简单的 ReAct Agent，而是一个以 EvidenceGraph 为核心工作记忆的代码仓库分析 Agent。

## 项目目标

构建一个面向代码仓库问答与分析的多 Agent 系统，将高层推理、问题分解、仓库调查和证据沉淀拆分为不同职责层。

## 架构概览

系统当前采用三层分工，但中间层已经从旧的 `FileReaderAgent` 路线演进为 `AnalyzerAgent + InvestigatorAgent`：

- `MainAgent` 负责高层推理、调查调度和最终回答。
- `AnalyzerAgent` 负责基于 `repo_profile.md` 和历史报告拆解高层调查任务。
- `InvestigatorAgent` 负责真正接触 repo tools，在预算约束下执行子调查并生成子报告。
- `.cache/repo-agent/` 负责持久化 `repo_profile.md` 和历史调查报告。

## Agent 分工

- `MainAgent` 只处理 evidence，不直接处理 repo 细节。
- `AnalyzerAgent` 负责问题分解、仓库画像维护和子报告综合，不直接深读代码。
- `InvestigatorAgent` 负责围绕单个子问题调用 `read_repo_tree / find_text / trace_symbol / read_file`，不直接写入 `EvidenceGraph`。

## EvidenceGraph 设计

`EvidenceGraph` 使用 append-only 结构保存被主 Agent 采纳的结论，并通过 `based_on` 关系维护高层证据链。

## Repo Cache

仓库级工作记忆持久化在目标仓库下的 `.cache/repo-agent/`：

- `repo_profile.md`：AnalyzerAgent 使用的自然语言仓库画像。
- `reports/`：历史 `InvestigationReport` 的 Markdown 缓存。

## Tool 权限边界

- `read_repo_tree`、`find_text`、`trace_symbol`、`read_file` 仅供 `InvestigatorAgent` 使用。
- `derive_claim` 仅供 `MainAgent` 使用。

## 使用方式

当前仓库以设计和基础实现为主。`InvestigatorAgent` 已经支持自动多轮 tool-calling；`AnalyzerAgent` 和 `MainAgent` 的完整主流程仍在继续补齐。

当前主设计文档是：

- `repo-agent_design_profile_reports_cache.md`

## 开发路线

优先继续完成 `AnalyzerAgent`、`request_investigation` / `request_subinvestigation` 的接线，以及 `MainAgent` 主循环。
