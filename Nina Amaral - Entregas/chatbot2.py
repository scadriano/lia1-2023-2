import os
from dotenv import load_dotenv
import openai

# Carregando variáveis de ambiente a partir do arquivo .env
load_dotenv()

# Obtendo a chave da API a partir das variáveis de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para gerar texto a partir do modelo de linguagem
def gera_texto(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response['choices'][0]['message']['content'].strip()

# Função principal do programa em Python
def main():
    print("\nBem-vindo ao GPT-4 Chatbot!")
    print("(Digite 'sair' a qualquer momento para encerrar o chat)")

    # Inicializando a lista de mensagens com a mensagem de boas-vindas
    messages = [{"role": "system", "content": "Você é um chatbot."},
                {"role": "user", "content": "Oi!"}]

    while True:
        # Coletando a pergunta digitada pelo usuário
        user_message = input("\nVocê: ").strip()

        # Verificando se o usuário deseja sair
        if user_message.lower() == "sair":
            break

        # Adicionando a mensagem do usuário à lista de mensagens
        messages.append({"role": "user", "content": user_message})

        # Obtendo a resposta do modelo
        chatbot_response = gera_texto(messages)

        # Adicionando a resposta do chatbot à lista de mensagens
        messages.append({"role": "assistant", "content": chatbot_response})

        # Imprimindo a resposta do chatbot
        print(f"\nChatbot: {chatbot_response}")

# Executando o programa
if __name__ == "__main__":
    main()