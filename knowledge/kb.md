# DevOps KB

## ECS cannot connect to RDS
Common causes:
- Security group inbound on RDS does not allow ECS task ENI SG on port 5432/3306.
- Subnets are in different VPCs with no peering / routing.
- NACL blocks traffic.
- Wrong DNS endpoint or private DNS not enabled.

## Terraform VPC peering issues
Common causes:
- Missing route table entries in both VPCs.
- SG rules allow only local CIDR.
- Peering exists but not accepted, or wrong region/account.

## CrashLoopBackOff
Common causes:
- App exits due to config/secret missing.
- Liveness probe too aggressive.
- Image tag wrong or cannot pull.
