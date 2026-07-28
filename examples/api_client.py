from api.python_client import GLMClient
client=GLMClient("http://localhost:8000")
token=client.register("developer@example.com","replace-this-with-a-strong-password")
client.close()
authenticated=GLMClient("http://localhost:8000",token)
print(authenticated.chat([{"role":"user","content":"Hello in English and Yoruba."}]))
authenticated.close()
