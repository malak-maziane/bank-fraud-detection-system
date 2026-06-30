-- Initialize Hive Metastore DB user and database
-- Creates the 'hive' role and 'metastore' database
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'hive') THEN
      CREATE ROLE hive LOGIN PASSWORD 'hive_password';
   END IF;
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metastore') THEN
      PERFORM dblink_exec('dbname=postgres', 'CREATE DATABASE metastore');
   END IF;
EXCEPTION WHEN others THEN
   -- ignore
END
$$;

GRANT ALL PRIVILEGES ON DATABASE metastore TO hive;
