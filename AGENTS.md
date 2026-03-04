# Repository Working Rules

1. Inner path (`/home/jonas/Drone/Drone`) is the primary source of truth.
2. Before the user confirms a change is valid, do not modify outer-layer files under:
   - `/home/jonas/Drone/source`
   - `/home/jonas/Drone/scripts`
3. During implementation and testing, make changes only in the inner repository first.
4. Sync changes from inner to outer only after explicit user confirmation.
5. Whenever environment settings are modified, add/update inline comments in the config/code and synchronize the same change summary into the environment README.
