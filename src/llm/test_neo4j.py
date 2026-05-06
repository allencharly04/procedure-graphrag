"""Smoke test: confirm Python can connect to Neo4j and run a query."""
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "assemblyrag2026"


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print("[*] Connection verified")

        with driver.session() as session:
            # Test 1: basic query
            result = session.run("RETURN 1 AS ok").single()
            assert result["ok"] == 1
            print("[*] RETURN 1 works")

            # Test 2: APOC plugin loaded?
            try:
                version = session.run("RETURN apoc.version() AS v").single()
                print("[*] APOC version:", version["v"])
            except Exception as e:
                print("[!] APOC not available:", str(e)[:120])

            # Test 3: write + read (and clean up)
            session.run("CREATE (:SmokeTest {ts: timestamp()})")
            count = session.run("MATCH (n:SmokeTest) RETURN count(n) AS c").single()["c"]
            print("[*] Created SmokeTest node, count:", count)
            session.run("MATCH (n:SmokeTest) DELETE n")
            after = session.run("MATCH (n:SmokeTest) RETURN count(n) AS c").single()["c"]
            print("[*] After delete, count:", after)

        print("[OK] Neo4j Python smoke test passed")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
