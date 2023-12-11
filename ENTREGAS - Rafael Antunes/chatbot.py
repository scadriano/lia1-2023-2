import os   
from dotenv import load_dotenv
import openai

# Carregando variáveis de ambiente a partir do arquivo .env
load_dotenv()

# Obtendo a chave da API a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para gerar texto a partir do modelo de linguagem
def gera_texto(texto):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=texto,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response['choices'][0]['text'].strip()

    # Função principal do programa em Python
def main():
    print("\nBem-vindo ao GPT-4 Chatbot!")
    print("(Digite 'sair' a qualquer momento para encerrar o chat)")

    while True:
        # Coletando a pergunta digitada pelo usuário
        user_message = input("\nVocê: ").strip()

        # Verificando se o usuário deseja sair
        if user_message.lower() == "sair":
            break

        # Adicionando a mensagem do usuário ao prompt
        gpt4_prompt = f"Usuário: {user_message}\nChatbot:"

        # Obtendo a resposta do modelo
        chatbot_response = gera_texto(gpt4_prompt)

        # Imprimindo a resposta do chatbot
        print(f"\nChatbot: {chatbot_response}")

# Executando o programa
if __name__ == "__main__":
    main()