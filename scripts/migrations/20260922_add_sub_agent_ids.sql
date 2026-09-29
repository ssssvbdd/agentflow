-- Agent-as-Tool: persist the explicit child-agent allowlist.
-- Run once for an existing MySQL deployment. New databases receive the
-- column from AgentTable through SQLModel.metadata.create_all().
ALTER TABLE agent
    ADD COLUMN sub_agent_ids JSON NOT NULL DEFAULT (JSON_ARRAY());
