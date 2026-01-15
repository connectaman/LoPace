# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do NOT** open a public issue

Please do not report security vulnerabilities through public GitHub issues. This helps protect the security of users.

### 2. **Contact Us Privately**

Please report security vulnerabilities by emailing:

- **Email**: [Your Security Email]
- **Subject**: `[LoPace Security] <Brief Description>`

### 3. **Include Details**

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability

### 4. **Response Timeline**

- **Initial Response**: Within 48 hours
- **Update**: Within 7 days with assessment
- **Resolution**: As quickly as possible, depending on complexity

### 5. **Disclosure Policy**

- We will acknowledge receipt of your vulnerability report
- We will keep you informed of our progress in resolving the issue
- We will notify you when the vulnerability is fixed
- We will credit you in the security advisory (unless you prefer otherwise)

### 6. **What to Report**

Please report security vulnerabilities including, but not limited to:

- Remote code execution (RCE)
- SQL injection
- Cross-site scripting (XSS)
- Authentication/Authorization bypasses
- Denial of service (DoS) vulnerabilities
- Data exposure or information leakage
- Cryptographic vulnerabilities

### 7. **What NOT to Report**

Please do not report:

- Missing security headers (unless they lead to a vulnerability)
- Missing best practices (unless they lead to a vulnerability)
- Theoretical issues without proof of concept
- Issues requiring physical or local access
- Issues requiring social engineering
- Missing or insufficient rate limiting (unless severe)
- Self-XSS
- Content spoofing / text injection issues without showing an attack vector
- Issues related to software dependencies (please report to upstream projects)

## Security Best Practices

When using LoPace:

1. **Keep Dependencies Updated**: Regularly update your dependencies
   ```bash
   pip install --upgrade lopace
   ```

2. **Verify Compressed Data**: Always verify decompressed data matches original
   ```python
   decompressed = compressor.decompress(compressed, method)
   assert decompressed == original  # Verify losslessness
   ```

3. **Use Strong Authentication**: If integrating with a service, use secure authentication

4. **Validate Input**: Always validate and sanitize input data before compression

5. **Handle Errors Gracefully**: Implement proper error handling in production code

## Acknowledgments

We appreciate the security researchers who help keep LoPace secure. Contributors who report valid security vulnerabilities will be:

- Listed in our security acknowledgments
- Credited in release notes
- Invited to contribute (if interested)

Thank you for helping keep LoPace and its users safe!