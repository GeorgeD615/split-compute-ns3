#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"

#include <fstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FileTransferSim");

Time g_startTime;

class FileReceiverApp : public Application
{
public:
    FileReceiverApp() : m_socket(0), m_expectedSize(0) {}

    void Setup(uint16_t port, const std::string& outputFile)
    {
        m_port = port;
        m_outputFile = outputFile;
    }

    void SetExpectedSize(uint32_t size)
    {
        m_expectedSize = size;
    }

    void StopManually()
    {
        StopApplication();
    }

private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
        m_socket->Listen();
        m_socket->SetAcceptCallback(
            MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
            MakeCallback(&FileReceiverApp::HandleAccept, this));
    }

    void HandleAccept(Ptr<Socket> socket, const Address&)
    {
        socket->SetRecvCallback(MakeCallback(&FileReceiverApp::HandleRead, this));
        m_socket = socket;
    }

    void HandleRead(Ptr<Socket> socket)
    {
        Ptr<Packet> packet;
        while ((packet = socket->Recv()))
        {
            std::vector<uint8_t> data(packet->GetSize());
            packet->CopyData(data.data(), data.size());
            m_buffer.insert(m_buffer.end(), data.begin(), data.end());

            if (m_expectedSize > 0 && m_buffer.size() >= m_expectedSize)
            {
                StopApplication();
                Simulator::Stop();
            }
        }
    }

    double ReadPreviousLatency(const std::string& filename) {
        std::ifstream infile(filename);
        double totalLatency = 0.0;

        if (infile) {
            infile >> totalLatency;
        }

        return totalLatency;
    }

    void LogCumulativeLatency(double latencySeconds, const std::string& filename) {
        double currentTotal = ReadPreviousLatency(filename);
        double newTotal = currentTotal + latencySeconds;

        std::ofstream outfile(filename, std::ios::trunc);
        if (outfile) {
            outfile << newTotal << std::endl;
        }
    }

    void StopApplication() override
    {
        Time latency = Simulator::Now() - g_startTime;
        LogCumulativeLatency(latency.GetSeconds(), "files/latency_transfer.log");


        NS_LOG_UNCOND(">>> Запись " << m_buffer.size() << " байт в файл " << m_outputFile);
        std::ofstream out(m_outputFile, std::ios::binary);
        out.write(reinterpret_cast<const char*>(m_buffer.data()), m_buffer.size());

        if (m_expectedSize > 0 && m_buffer.size() != m_expectedSize)
        {
            NS_LOG_UNCOND(">>> ВНИМАНИЕ: Получено " << m_buffer.size()
                          << " байт из ожидаемых " << m_expectedSize << ". Данные повреждены!");
        }

        if (m_socket) m_socket->Close();
    }

    Ptr<Socket> m_socket;
    uint16_t m_port;
    std::string m_outputFile;
    std::vector<uint8_t> m_buffer;
    uint32_t m_expectedSize;
};

class FileSenderApp : public Application
{
public:
    FileSenderApp() : m_socket(0) {}

    void Setup(Address address, const std::string& inputFile, uint32_t, Time, Ptr<FileReceiverApp> receiver)
    {
        m_peerAddress = address;
        m_receiverApp = receiver;

        std::ifstream file(inputFile, std::ios::binary);
        if (!file) NS_FATAL_ERROR("Cannot open input file: " << inputFile);

        m_data.assign(std::istreambuf_iterator<char>(file), {});
        receiver->SetExpectedSize(m_data.size());
    }

private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->Connect(m_peerAddress);
        m_socket->SetConnectCallback(
            MakeCallback(&FileSenderApp::ConnectionSucceeded, this),
            MakeCallback(&FileSenderApp::ConnectionFailed, this));
    }

    uint32_t m_sendIndex;

    void SendData()
    {
        while (m_sendIndex < m_data.size())
        {
            uint32_t chunkSize = std::min<uint32_t>(1024, m_data.size() - m_sendIndex);
            int sent = m_socket->Send(reinterpret_cast<const uint8_t*>(&m_data[m_sendIndex]), chunkSize, 0);

            if (sent <= 0)
            {
                break;
            }

            m_sendIndex += sent;
        }

        if (m_sendIndex >= m_data.size())
        {
            NS_LOG_UNCOND(">>> Отправлено байт: " << m_sendIndex);
        }
    }

    void HandleSend(Ptr<Socket>, uint32_t)
    {
        SendData();
    }

    void ConnectionSucceeded(Ptr<Socket>)
    {
        NS_LOG_UNCOND(">>> Размер данных к отправке: " << m_data.size());
        g_startTime = Simulator::Now();
        NS_LOG_UNCOND(">>> TCP подключение успешно. Отправка...");

        m_sendIndex = 0;
        m_socket->SetSendCallback(MakeCallback(&FileSenderApp::HandleSend, this));
        SendData();
    }

    void RetryConnect()
    {
        NS_LOG_UNCOND(">>> Повторная попытка TCP подключения...");
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->Connect(m_peerAddress);
        m_socket->SetConnectCallback(
            MakeCallback(&FileSenderApp::ConnectionSucceeded, this),
            MakeCallback(&FileSenderApp::ConnectionFailed, this));
    }

    void ConnectionFailed(Ptr<Socket>)
    {
        NS_LOG_UNCOND(">>> Ошибка TCP подключения.");
        Simulator::Schedule(Seconds(1.0), &FileSenderApp::RetryConnect, this);
    }

    void StopApplication() override
    {
        if (m_socket) m_socket->Close();
    }

    Ptr<Socket> m_socket;
    Address m_peerAddress;
    std::vector<uint8_t> m_data;
    Ptr<FileReceiverApp> m_receiverApp;
};

void PhyRxMonitor(std::string, Ptr<const Packet>, uint16_t, WifiTxVector,
                  MpduInfo, SignalNoiseDbm signalNoise, uint16_t)
{
    double rssi = signalNoise.signal;
    double snr = rssi - signalNoise.noise;

    static std::ofstream out("files/channel_metrics.log", std::ios::app);
    out << Simulator::Now().GetSeconds()
        << ",RSSI=" << rssi
        << ",SNR=" << snr << std::endl;
}

int main(int argc, char* argv[])
{
    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor;

    double distance = 30.0, interval = 0.1;
    uint32_t packetSize = 512;
    std::string inputFile, outputFile, phyStandard = "802.11g";
    double m0 = 1.0, m1 = 1.0, m2 = 1.0;

    CommandLine cmd;
    cmd.AddValue("distance", "Distance between nodes", distance);
    cmd.AddValue("phyStandard", "WiFi PHY standard", phyStandard);
    cmd.AddValue("inputFile", "Path to input file", inputFile);
    cmd.AddValue("outputFile", "Path to save received file", outputFile);
    cmd.AddValue("packetSize", "Packet size in bytes", packetSize);
    cmd.AddValue("interval", "Interval between packets", interval);
    cmd.AddValue("m0", "Nakagami m0 parameter", m0);
    cmd.AddValue("m1", "Nakagami m1 parameter", m1);
    cmd.AddValue("m2", "Nakagami m2 parameter", m2);
    cmd.Parse(argc, argv);

    if (inputFile.empty() || outputFile.empty()) {
        NS_FATAL_ERROR("inputFile и outputFile обязательные параметры.");
    }

    NodeContainer nodes;
    nodes.Create(2);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.AddPropagationLoss("ns3::NakagamiPropagationLossModel",
                               "Distance1", DoubleValue(80.0),
                               "Distance2", DoubleValue(250.0),
                               "m0", DoubleValue(m0),
                               "m1", DoubleValue(m1),
                               "m2", DoubleValue(m2));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    phy.Set("RxGain", DoubleValue(5.0));
    phy.Set("TxPowerStart", DoubleValue(25.0));
    phy.Set("TxPowerEnd", DoubleValue(25.0));

    WifiHelper wifi;
    wifi.SetStandard((phyStandard == "802.11n") ? WIFI_STANDARD_80211n : WIFI_STANDARD_80211g);

    WifiMacHelper mac;
    Ssid ssid = Ssid("file-transfer");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevice = wifi.Install(phy, mac, nodes.Get(0));

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, nodes.Get(1));

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(distance),
                                  "GridWidth", UintegerValue(2),
                                  "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign({staDevice.Get(0), apDevice.Get(0)});

    uint16_t port = 9000;

    Ptr<FileReceiverApp> receiver = CreateObject<FileReceiverApp>();
    receiver->Setup(port, outputFile);
    nodes.Get(1)->AddApplication(receiver);
    receiver->SetStartTime(Seconds(1.0));

    Ptr<FileSenderApp> sender = CreateObject<FileSenderApp>();
    sender->Setup(InetSocketAddress(interfaces.GetAddress(1), port), inputFile, packetSize, Seconds(interval), receiver);
    nodes.Get(0)->AddApplication(sender);
    sender->SetStartTime(Seconds(2.0));

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx", MakeCallback(&PhyRxMonitor));

    monitor = flowHelper.InstallAll();
    Simulator::Run();

    monitor->CheckForLostPackets();
    auto classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    auto stats = monitor->GetFlowStats();

    std::ofstream out("files/throughput.log");
    for (const auto& flow : stats)
    {
        double duration = flow.second.timeLastRxPacket.GetSeconds() - flow.second.timeFirstTxPacket.GetSeconds();
        double throughput = (flow.second.rxBytes * 8.0) / duration / 1e6;
        out << "FlowID: " << flow.first << ", Throughput: " << throughput << " Mbps" << std::endl;
    }

    Simulator::Destroy();
    return 0;
}
