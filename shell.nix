{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = [
    pkgs.just # Just
    pkgs.uv # Python package manager
    pkgs.python311 # Python 3.11
  ];

  # Shell hook to set up environment
  shellHook = ''
    just install
  '';
}
