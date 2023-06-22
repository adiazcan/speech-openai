using Azure;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI.ChatCompletion;

var configuration = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", false)
    .AddJsonFile("appsettings.development.json", false)
    .Build();

string openAIKey = configuration["OpenAIKey"];
string openAIEndpoint = configuration["OpenAIEndpoint"];
string openAIModel = configuration["OpenAIModel"];

string speechKey = configuration["SpeechKey"];
string speechRegion = configuration["SpeechRegion"];


using ILoggerFactory loggerFactory = LoggerFactory.Create(builder =>
{
    builder.AddConfiguration(configuration.GetSection("Logging"))
        .AddConsole()
        .AddDebug();
});

var kernel = Kernel.Builder
    .WithAzureChatCompletionService(openAIModel, openAIEndpoint, openAIKey)
    .WithLogger(loggerFactory.CreateLogger<IKernel>())
    .Build();

await ChatWithOpenAI();

async Task AskOpenAI(string question)
{
    var systemMessage = @"ChatBot can have a conversation with you about any topic.
        It can give explicit instructions or say 'I don't know' if it does not have an answer.";
    
    var chatGPT = kernel.GetService<IChatCompletion>();
    var chat = (OpenAIChatHistory)chatGPT.CreateNewChat(systemMessage);
    chat.AddUserMessage(question);

    var reply = await chatGPT.GenerateMessageAsync(chat, new ChatRequestSettings());
    Console.WriteLine($"Azure OpenAI response: {reply}");

    var speechConfig = SpeechConfig.FromSubscription(speechKey, speechRegion);
    speechConfig.SpeechSynthesisVoiceName = "es-ES-DarioNeural";

    var audioOutputConfig = AudioConfig.FromDefaultSpeakerOutput();

    using var synthesizer = new SpeechSynthesizer(speechConfig, audioOutputConfig);
    var speechSynthesisResult = await synthesizer.SpeakTextAsync(reply);

    if (speechSynthesisResult.Reason == ResultReason.SynthesizingAudioCompleted)
    {
        Console.WriteLine($"Speech synthesized to speaker for text: [{reply}]");
    }
    else if (speechSynthesisResult.Reason == ResultReason.Canceled)
    {
        var cancellationDetails = SpeechSynthesisCancellationDetails.FromResult(speechSynthesisResult);
        Console.WriteLine($"Speech synthesis canceled: {cancellationDetails.Reason}");

        if (cancellationDetails.Reason == CancellationReason.Error)
        {
            Console.WriteLine($"Error details: {cancellationDetails.ErrorDetails}");
        }
    }
}
    
async Task ChatWithOpenAI()
{
    // Should be the locale for the speaker's language.
    var speechConfig = SpeechConfig.FromSubscription(speechKey, speechRegion);        
    speechConfig.SpeechRecognitionLanguage = "es-ES";

    using var audioConfig = AudioConfig.FromDefaultMicrophoneInput();
    using var speechRecognizer = new SpeechRecognizer(speechConfig, audioConfig);
    var conversationEnded = false;

    while(!conversationEnded)
    {
        Console.WriteLine("Azure OpenAI is listening. Say 'Stop' or press Ctrl-Z to end the conversation.");

        // Get audio from the microphone and then send it to the TTS service.
        var speechRecognitionResult = await speechRecognizer.RecognizeOnceAsync();           

        switch (speechRecognitionResult.Reason)
        {
            case ResultReason.RecognizedSpeech:
                if (speechRecognitionResult.Text == "Stop.")
                {
                    Console.WriteLine("Conversation ended.");
                    conversationEnded = true;
                }
                else
                {
                    Console.WriteLine($"Recognized speech: {speechRecognitionResult.Text}");
                    await AskOpenAI(speechRecognitionResult.Text);
                }
                break;
            case ResultReason.NoMatch:
                Console.WriteLine($"No speech could be recognized: ");
                break;
            case ResultReason.Canceled:
                var cancellationDetails = CancellationDetails.FromResult(speechRecognitionResult);
                Console.WriteLine($"Speech Recognition canceled: {cancellationDetails.Reason}");
                if (cancellationDetails.Reason == CancellationReason.Error)
                {
                    Console.WriteLine($"Error details={cancellationDetails.ErrorDetails}");
                }
                break;
        }
    }
}