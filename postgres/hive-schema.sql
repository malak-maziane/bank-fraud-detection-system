-- Initialize Hive Metastore database role and grants
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'hive') THEN
    CREATE ROLE hive LOGIN PASSWORD 'hive_password';
  END IF;
END
$$;

GRANT ALL PRIVILEGES ON DATABASE metastore TO hive;
ALTER DATABASE metastore OWNER TO hive;
