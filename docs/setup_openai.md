## OpenAI API Key
To use the OpenAI API, ensure `OPENAI_API_KEY` is set in your environment variables before proceeding. 

##### **For Windows Users:**
```cmd
setx OPENAI_API_KEY <your_openai_api_key>
```
To check if the key is set, run:
```cmd
echo %OPENAI_API_KEY%
```

##### **For Linux and MacOS Users:**
Using a text editor of your choice (`nano` in this example), open the `.bashrc` file in your home directory:
```bash
nano ~/.bashrc
```
Add the following line:
```bash
export OPENAI_API_KEY=<your_openai_api_key>
```
Save the file and exit the editor. Apply the changes with:
```bash
source ~/.bashrc
```
To check if the key is set, run:
```bash
echo $OPENAI_API_KEY
```