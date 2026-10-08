{
  description = "rag-mcp: local hybrid RAG (SQLite FTS5 + candle embeddings) as an MCP server";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      packages.${system} = rec {
        rag-mcp = pkgs.rustPlatform.buildRustPackage {
          pname = "rag-mcp";
          version = "0.1.0";
          src = pkgs.lib.cleanSource ./.;
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl pkgs.sqlite ];
          # Tests need the HF model (~450MB); run `cargo test` in devShell.
          doCheck = false;
        };
        default = rag-mcp;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.rustc
          pkgs.cargo
          pkgs.rust-analyzer
          pkgs.clippy
          pkgs.rustfmt
          pkgs.pkg-config
          pkgs.openssl
          pkgs.sqlite
          pkgs.poppler-utils
        ];
        # Be gentle on small laptops: fewer parallel codegen jobs.
        CARGO_BUILD_JOBS = "4";
        RUST_BACKTRACE = "1";
      };
    };
}
