# Security Policy

## Supported Versions

Security fixes are released as a new version of PyPatchMatch on PyPI. Only the latest
release is supported; older versions are not updated.

## Reporting a Vulnerability

Please report vulnerabilities privately through GitHub's
[vulnerability reporting](https://github.com/mauwii/PyPatchMatch/security/advisories/new),
not in a public issue.

## OpenCV

The wheels on PyPI contain a statically linked, core-only build of OpenCV; its version
is listed in the CycloneDX SBOM in `sboms/opencv.cdx.json` of each wheel's
`.dist-info`. A vulnerability in OpenCV core is fixed by a new PyPatchMatch release
with an updated OpenCV. Installations built from the source distribution link the
OpenCV of the system and get its fixes from there.
