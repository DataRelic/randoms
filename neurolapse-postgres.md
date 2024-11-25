### install sudo apt install -y git build-essential libpq-dev postgresql-server-dev-14
### add support for vectors
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector/
make
sudo make install ```


CREATE DATABASE neurolapse_db
    WITH 
    OWNER = neurolapse_dev
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Grant privileges to neurolapse_dev
\c neurolapse_db
GRANT ALL PRIVILEGES ON DATABASE neurolapse_db TO neurolapse_dev;
GRANT ALL PRIVILEGES ON SCHEMA public TO neurolapse_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO neurolapse_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO neurolapse_dev;


