mkdir -p ~/.streamlit/

# - name: 'Create env file'
# run: |
#     echo "${{ secrets.ENV_FILE }}" > .env

echo "\
[server]\n\
headless=true\n\
port=$PORT\n\
enableCORS=false\n\
\n\
" > ~/.streamlit/config.toml