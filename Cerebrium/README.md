# Cerebrium

## Steps

1. **Create a Cerebrium Account**

   * You can get 30 free credits to run things on their platform.

2. **Install the Cerebrium CLI**

   ```bash
   pip install cerebrium --upgrade
   ```

3. **Create a New App**

   ```bash
   cerebrium init my-first-project
   ```

4. **Change to Project Directory**

   ```bash
   cd my-first-project
   ```

   * Add all the files from the `/Cerebrium_API` folder into this project directory.

5. **Deploy Your App**

   ```bash
   cerebrium deploy
   ```

   * Follow the prompts to sign in and deploy to your account.

6. **Test**

   * Use the `bench.py` script to test each GPU using the same input and code.
   * To switch GPUs, change the GPU in the project config file, then re-deploy.
   * For my data, I only tested the GPUs available to free plan users: **T4, L4, L40, A10**.


