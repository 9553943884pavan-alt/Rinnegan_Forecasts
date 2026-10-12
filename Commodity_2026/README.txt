Commodity Open 2026: kit and data pack

commodity-open-kit-2.6.2.zip        the participant kit: rules 2.6, scorer, template, notebooks, tools
commodity-open-pack-v2.0.1.tar.zst  the data pack, data to the close of Fri 25 Sep 2026
MANIFEST.json                       every table of the pack with its sha256 and row count
SHA256SUMS                          checksums of the three files above
update-from-v2.0/                   only for teams that already hold pack v2.0, kit 2.6, the kit of 2
                                    October or kit 2.6.1 (the files of 1, 2 and 5 October); read its
                                    README.txt

Start here:
  sha256sum -c SHA256SUMS
  unzip commodity-open-kit-2.6.2.zip && tar --zstd -xf commodity-open-pack-v2.0.1.tar.zst
  then follow commodity-open-kit/README.md
tar needs a zstd program. If it stops with "zstd: Cannot exec: No such file or directory" (GNU tar) or
"unable to run program "zstd -d -qq"" (macOS), install one: brew install zstd (macOS), or
sudo apt install zstd unzip (Ubuntu, WSL).

The kit is Apache-2.0 (see its LICENSE and README.md). The pack and the second data file are for your
team's use in Commodity Open 2026 only: until the verdicts are sent they may not be shared outside the
team or redistributed in any form.

Questions: contact@commodity-open.xyz, quoting your ticket number.
